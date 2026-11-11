"""Agents: the only two places an LLM is used.

Extraction agent - turns unstructured invoice text into a validated Invoice.
  Controls: schema-constrained output, retry with error feedback, fallback model, grounding score.
Drafting agent   - writes the controller memo and vendor email from *findings only* (never raw
  invoice) and falls back to a deterministic template, so a reviewer always gets a draft.
"""
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from importlib import import_module

from langchain_core.messages import HumanMessage, SystemMessage

from .grounding import check_grounding
from .schemas import Drafts, ExtractionResult, InvoiceExtraction, Settings

log = logging.getLogger(__name__)

# extraction agent
SYSTEM_PROMPT = (
    "You extract data from supplier invoices. Copy values exactly as printed. "
    "Never calculate, correct, or infer numbers. If a field is absent, use an empty string or 0."
)


def load_document(path: Path) -> str:
    """Read an invoice from .txt or a text-based .pdf."""
    if path.suffix.lower() == ".pdf":
        PdfReader = import_module("pypdf").PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        if len(text.strip()) < 40:
            raise ValueError(f"{path.name}: no extractable text (scanned PDF?). OCR is required first.")
        return text
    return path.read_text(encoding="utf-8")


# orchestration
def extract_invoice(
    raw_text: str,
    llms: Sequence[tuple[str, Any]],
    settings: Settings,
) -> ExtractionResult:
    """Try each (name, llm) in order with retries; return the best result or a failure record."""
    best: ExtractionResult | None = None
    attempts, last_error = 0, None

    for name, llm in llms:
        extractor = llm.with_structured_output(InvoiceExtraction)
        feedback = ""
        for _ in range(settings.max_attempts_per_model):
            attempts += 1
            try:
                out = extractor.invoke(
                    [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=f"Invoice text:\n---\n{raw_text}\n---{feedback}"),
                    ]
                )
                if isinstance(out, dict):
                    out = InvoiceExtraction.model_validate(out)
                invoice = out.to_domain()
                confidence, ungrounded = check_grounding(invoice, raw_text)
                result = ExtractionResult(
                    invoice=invoice, confidence=confidence, model_used=name,
                    attempts=attempts, ungrounded=ungrounded,
                )
                if best is None or confidence > best.confidence:
                    best = result
                if confidence >= settings.min_extraction_confidence:
                    return result
                feedback = (
                    "\n\nThese values do NOT appear in the invoice text: "
                    f"{ungrounded}. Re-read the text and copy values exactly."
                )
                log.warning("Low-confidence extraction (%s, %.2f): %s", name, confidence, ungrounded)
            except Exception as exc:  # model down, invalid JSON, schema violation...
                last_error = f"{name}: {type(exc).__name__}: {exc}"
                feedback = f"\n\nYour previous output was invalid ({exc}). Return valid output only."
                log.warning("Extraction attempt failed: %s", last_error)

    if best is not None:
        return best.model_copy(update={"attempts": attempts})
    return ExtractionResult(attempts=attempts, error=last_error or "no model available")


# drafting agent
Drafter = Callable[[dict], tuple[Drafts, str]]

SYSTEM = (
    "You are a senior procurement audit consultant writing for a finance controller. "
    "Use ONLY the facts provided. Do not introduce any figure that is not in the facts. "
    "Be concise and specific."
)


def template_drafts(facts: dict) -> Drafts:
    bullets = "\n".join(f"- [{f['severity']}] {f['message']}" for f in facts["findings"]) or "- None"
    memo = (
        f"AUDIT MEMO - Invoice {facts['invoice_id']} ({facts['vendor_name']})\n"
        f"Recommendation: {facts['recommendation']} | Risk score: {facts['risk_score']} | "
        f"Amount at risk: {facts['amount_at_risk']}\n\nFindings:\n{bullets}"
    )
    email = (
        f"Dear {facts['vendor_name']} accounts team,\n\nWe reviewed invoice {facts['invoice_id']} against "
        f"contract terms and found the following issues:\n{bullets}\n\n"
        "Please issue a corrected invoice or supporting documentation.\n\nKind regards,\nAccounts Payable"
    )
    return Drafts(memo=memo, vendor_email=email)


def make_drafter(llm: Optional[Any] = None) -> Drafter:
    def draft(facts: dict) -> tuple[Drafts, str]:
        if llm is not None:
            try:
                out = llm.with_structured_output(Drafts).invoke(
                    [SystemMessage(content=SYSTEM), HumanMessage(content=f"Facts:\n{facts}")]
                )
                if isinstance(out, dict):
                    out = Drafts.model_validate(out)
                return out, "llm"
            except Exception as exc:
                log.warning("LLM drafting failed (%s); using template.", exc)
        return template_drafts(facts), "template"

    return draft
