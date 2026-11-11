"""Workflow: LangGraph workflow with fallback handling and a real human-in-the-loop pause.

    extract -> audit -> [clean & confident] -> post_to_erp
                     -> [anything else]     -> draft -> human_review (interrupt) -> finalize

The graph runs on a checkpointer, so a paused review survives until a controller resumes it.
"""
import hashlib
import json
import operator
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .agents import Drafter, extract_invoice, make_drafter
from .rules import amount_at_risk, recommend, risk_score, run_rules
from .schemas import Contract, ExtractionResult, Finding, Invoice, Severity, Settings


# tracing (append-only audit trail)
def write_audit_record(path: str, state: dict, final_status: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(state["raw_text"].encode()).hexdigest(),
        "invoice_id": (state.get("invoice") or {}).get("invoice_id"),
        "extraction": state.get("extraction_meta"),
        "findings": [f["code"] for f in state.get("findings", [])],
        "risk_score": state.get("risk_score"),
        "recommendation": state.get("recommendation"),
        "draft_source": state.get("draft_source"),
        "human_decision": state.get("human_decision"),
        "final_status": final_status,
        "trace": state.get("trace", []),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# the workflow graph
class AuditState(TypedDict, total=False):
    raw_text: str
    contract: dict
    known_invoice_ids: list[str]
    invoice: Optional[dict]
    extraction_confidence: float
    extraction_meta: dict
    findings: list[dict]
    risk_score: float
    amount_at_risk: str
    recommendation: str
    memo: str
    vendor_email: str
    draft_source: str
    human_decision: dict
    status: str
    trace: Annotated[list[str], operator.add]


ExtractFn = Callable[[str], ExtractionResult]


def build_graph(extract_fn: ExtractFn, drafter: Drafter, settings: Settings, checkpointer=None):
    """Dependencies are injected so the workflow is testable without a running LLM."""

    def extract_node(state: AuditState) -> dict:
        t0 = time.perf_counter()
        res = extract_fn(state["raw_text"])
        return {
            "invoice": res.invoice.model_dump(mode="json") if res.invoice else None,
            "extraction_confidence": res.confidence,
            "extraction_meta": res.model_dump(mode="json", exclude={"invoice"}),
            "trace": [f"extract: model={res.model_used} attempts={res.attempts} "
                      f"confidence={res.confidence} ({time.perf_counter() - t0:.1f}s)"],
        }

    def audit_node(state: AuditState) -> dict:
        conf = state["extraction_confidence"]
        if state["invoice"] is None:
            findings = [Finding(code="EXTRACTION_FAILED", severity=Severity.HIGH,
                                message="Automatic extraction failed with all models; manual keying required.")]
        else:
            findings = run_rules(
                Invoice.model_validate(state["invoice"]),
                Contract.model_validate(state["contract"]),
                state.get("known_invoice_ids", []),
                settings,
            )
            if conf < settings.min_extraction_confidence:
                bad = ", ".join(state["extraction_meta"]["ungrounded"])
                findings.append(Finding(code="LOW_EXTRACTION_CONFIDENCE", severity=Severity.MEDIUM,
                                        message=f"Extracted values not found in source text: {bad}."))
        rec = recommend(findings, conf, settings)
        return {
            "findings": [f.model_dump(mode="json") for f in findings],
            "risk_score": 1.0 if state["invoice"] is None else risk_score(findings),
            "amount_at_risk": str(amount_at_risk(findings)),
            "recommendation": rec,
            "trace": [f"audit: {len(findings)} findings -> {rec}"],
        }

    def route_after_audit(state: AuditState) -> str:
        return "post_to_erp" if state["recommendation"] == "APPROVE" else "draft"

    def post_node(state: AuditState) -> dict:
        status = "POSTED_TO_ERP (STRAIGHT_THROUGH)"
        write_audit_record(settings.audit_log_path, {**state, "human_decision": None}, status)
        return {"status": status, "trace": ["post: auto-approved, no exceptions"]}

    def draft_node(state: AuditState) -> dict:
        inv = state["invoice"] or {}
        facts = {
            "invoice_id": inv.get("invoice_id", "UNKNOWN"),
            "vendor_name": inv.get("vendor_name", "UNKNOWN"),
            "recommendation": state["recommendation"],
            "risk_score": state["risk_score"],
            "amount_at_risk": state["amount_at_risk"],
            "findings": [{"severity": f["severity"], "message": f["message"]} for f in state["findings"]],
        }
        drafts, source = drafter(facts)
        return {"memo": drafts.memo, "vendor_email": drafts.vendor_email, "draft_source": source,
                "trace": [f"draft: source={source}"]}

    def review_node(state: AuditState) -> dict:
        # Execution pauses here until a human resumes the graph. Keep this node side-effect free:
        # LangGraph re-runs it from the top on resume.
        decision = interrupt({
            "invoice_id": (state["invoice"] or {}).get("invoice_id"),
            "recommendation": state["recommendation"],
            "risk_score": state["risk_score"],
            "amount_at_risk": state["amount_at_risk"],
            "memo": state["memo"],
        })
        if decision.get("decision") not in {"APPROVE", "REJECT"}:
            raise ValueError("Reviewer decision must be APPROVE or REJECT.")
        return {"human_decision": decision, "trace": [f"review: {decision['decision']} by {decision.get('reviewer')}"]}

    def finalize_node(state: AuditState) -> dict:
        status = ("POSTED_TO_ERP (APPROVED_BY_REVIEWER)" if state["human_decision"]["decision"] == "APPROVE"
                  else "RETURNED_TO_VENDOR")
        write_audit_record(settings.audit_log_path, state, status)
        return {"status": status, "trace": [f"finalize: {status}"]}

    g = StateGraph(AuditState)
    g.add_node("extract", extract_node)
    g.add_node("audit", audit_node)
    g.add_node("post_to_erp", post_node)
    g.add_node("draft", draft_node)
    g.add_node("human_review", review_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("extract")
    g.add_edge("extract", "audit")
    g.add_conditional_edges("audit", route_after_audit, {"post_to_erp": "post_to_erp", "draft": "draft"})
    g.add_edge("post_to_erp", END)
    g.add_edge("draft", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


# wiring for real runs (Ollama)
def build_app(settings: Settings | None = None):
    from langchain_ollama import ChatOllama

    settings = settings or Settings()
    mk = lambda name: ChatOllama(model=name, temperature=0, base_url=settings.ollama_url)  # noqa: E731
    primary, fallback = mk(settings.primary_model), mk(settings.fallback_model)
    extract_fn = partial(
        extract_invoice,
        llms=[(settings.primary_model, primary), (settings.fallback_model, fallback)],
        settings=settings,
    )
    return build_graph(extract_fn, make_drafter(primary), settings), settings


def load_contract(path: Path) -> dict:
    return Contract.model_validate_json(path.read_text()).model_dump(mode="json")
