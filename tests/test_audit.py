"""All tests run without an LLM: a scripted FakeLLM stands in for Ollama."""
import json
import sys
import uuid
from decimal import Decimal
from functools import partial
from pathlib import Path

import pytest
from langgraph.types import Command

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from audit.agents import extract_invoice, make_drafter
from audit.grounding import check_grounding
from audit.rules import amount_at_risk, parse_payment_terms_days, recommend, risk_score, run_rules
from audit.schemas import Contract, InvoiceExtraction, LineItemExtraction, Settings
from audit.workflow import build_graph

# fixtures & fakes
class FakeStructured:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, _messages):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeLLM:
    """Stands in for ChatOllama: returns scripted structured outputs or raises scripted errors."""

    def __init__(self, responses):
        self._s = FakeStructured(responses)

    def with_structured_output(self, _schema):
        return self._s


@pytest.fixture
def settings(tmp_path):
    return Settings(audit_log_path=str(tmp_path / "audit.jsonl"))


@pytest.fixture
def contract():
    return Contract.model_validate_json((ROOT / "data/contracts/apex_msa.json").read_text())


def make_extraction(**over) -> InvoiceExtraction:
    base = dict(
        vendor_name="Apex Cloud Technologies Ltd.", invoice_id="INV-2026-9040", payment_terms="Net 45",
        items=[
            LineItemExtraction(description="Cloud Infrastructure Managed Services", quantity=40, unit_price=175.0, line_total=7000.0),
            LineItemExtraction(description="Enterprise Security Log Review", quantity=10, unit_price=190.0, line_total=1900.0),
        ],
        subtotal=8900.0, tax=890.0, grand_total=9790.0,
    )
    base.update(over)
    return InvoiceExtraction(**base)


@pytest.fixture
def clean_text():
    return (ROOT / "data/invoices/inv_clean.txt").read_text()


@pytest.fixture
def bad_text():
    return (ROOT / "data/invoices/inv_violations.txt").read_text()

# guardrails & controls (rules)
def codes(findings):
    return {f.code for f in findings}


def test_clean_invoice_has_no_findings(contract, settings):
    inv = make_extraction().to_domain()
    assert run_rules(inv, contract, [], settings) == []


def test_rate_ceiling_breach_and_exposure(contract, settings):
    inv = make_extraction(
        items=[make_extraction().items[0].model_copy(update={"unit_price": 220.0, "line_total": 8800.0})],
        subtotal=8800.0, tax=880.0, grand_total=9680.0,
    ).to_domain()
    f = run_rules(inv, contract, [], settings)
    breach = next(x for x in f if x.code == "RATE_CEILING_BREACH")
    assert breach.amount_at_risk == Decimal("1400.00")  # (220 - 185) * 40


def test_payment_terms_parsing_and_mismatch(contract, settings):
    assert parse_payment_terms_days("NET  30") == 30
    assert parse_payment_terms_days("Due on receipt") == 0
    assert parse_payment_terms_days("whenever") is None
    f = run_rules(make_extraction(payment_terms="Net 15").to_domain(), contract, [], settings)
    assert "PAYMENT_TERMS_MISMATCH" in codes(f)


def test_arithmetic_controls(contract, settings):
    f = run_rules(make_extraction(subtotal=9000.0, grand_total=9890.0).to_domain(), contract, [], settings)
    assert "SUBTOTAL_MISMATCH" in codes(f)
    f = run_rules(make_extraction(grand_total=10000.0).to_domain(), contract, [], settings)
    assert "GRAND_TOTAL_MISMATCH" in codes(f)


def test_duplicate_vendor_and_uncontracted(contract, settings):
    f = run_rules(make_extraction().to_domain(), contract, ["INV-2026-9040"], settings)
    assert "DUPLICATE_INVOICE" in codes(f)
    f = run_rules(make_extraction(vendor_name="Totally Different Corp").to_domain(), contract, [], settings)
    assert "VENDOR_MISMATCH" in codes(f)
    items = make_extraction().items + [make_extraction().items[0].model_copy(update={"description": "Concierge Gold Package"})]
    f = run_rules(make_extraction(items=items, subtotal=15900.0, tax=1590.0, grand_total=17490.0).to_domain(), contract, [], settings)
    assert "UNCONTRACTED_ITEM" in codes(f)


def test_vendor_suffix_and_punctuation_tolerated(contract, settings):
    f = run_rules(make_extraction(vendor_name="APEX CLOUD TECHNOLOGIES LIMITED").to_domain(), contract, [], settings)
    assert "VENDOR_MISMATCH" not in codes(f)


def test_scoring_and_recommendation(contract, settings):
    bad = run_rules(make_extraction(payment_terms="Net 15").to_domain(), contract, [], settings)
    assert 0 < risk_score(bad) < 1
    assert risk_score([]) == 0.0
    assert recommend([], 1.0, settings) == "APPROVE"
    assert recommend([], 0.5, settings) == "MANUAL_REVIEW"  # low-confidence extraction never auto-approves
    assert recommend(bad, 1.0, settings) == "MANUAL_REVIEW"
    dup = run_rules(make_extraction().to_domain(), contract, ["INV-2026-9040"], settings)
    assert recommend(dup, 1.0, settings) == "REJECT"
    assert amount_at_risk(dup) == Decimal("9790.00")

# extraction agent & grounding
def test_grounding_full_and_hallucinated(clean_text):
    conf, bad = check_grounding(make_extraction().to_domain(), clean_text)
    assert conf == 1.0 and bad == []
    conf, bad = check_grounding(make_extraction(tax=999.0).to_domain(), clean_text)
    assert conf < 1.0 and "tax=999.0" in bad


def test_retry_after_invalid_output(clean_text, settings):
    llm = FakeLLM([ValueError("bad json"), make_extraction()])
    res = extract_invoice(clean_text, [("primary", llm)], settings)
    assert res.invoice is not None and res.attempts == 2 and res.model_used == "primary"


def test_fallback_model_used_when_primary_down(clean_text, settings):
    primary = FakeLLM([ConnectionError("ollama down")] * 2)
    fallback = FakeLLM([make_extraction()])
    res = extract_invoice(clean_text, [("primary", primary), ("fallback", fallback)], settings)
    assert res.model_used == "fallback" and res.confidence == 1.0


def test_total_failure_is_reported_not_raised(clean_text, settings):
    res = extract_invoice(clean_text, [("m", FakeLLM([ConnectionError("down")] * 2))], settings)
    assert res.invoice is None and "ConnectionError" in res.error


def test_low_confidence_result_is_kept_when_nothing_better(clean_text, settings):
    llm = FakeLLM([make_extraction(tax=999.0, grand_total=9899.0)] * 2)
    res = extract_invoice(clean_text, [("m", llm)], settings)
    assert res.invoice is not None and res.confidence < settings.min_extraction_confidence

# workflow (human-in-the-loop)
def run(settings, contract, text, llm_responses, known=()):
    extract_fn = partial(extract_invoice, llms=[("fake", FakeLLM(llm_responses))], settings=settings)
    app = build_graph(extract_fn, make_drafter(None), settings)  # template drafter
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    app.invoke({"raw_text": text, "contract": contract.model_dump(mode="json"),
                "known_invoice_ids": list(known), "trace": []}, cfg)
    return app, cfg


def log_lines(settings):
    return [json.loads(l) for l in open(settings.audit_log_path)]


def test_clean_invoice_is_straight_through(settings, contract, clean_text):
    app, cfg = run(settings, contract, clean_text, [make_extraction()])
    snap = app.get_state(cfg)
    assert not snap.next and snap.values["status"].startswith("POSTED_TO_ERP (STRAIGHT")
    assert log_lines(settings)[0]["final_status"].startswith("POSTED")


def test_violation_pauses_for_human_then_approves(settings, contract, bad_text):
    bad = make_extraction(
        invoice_id="INV-2026-9041", payment_terms="Net 15",
        items=[make_extraction().items[0], make_extraction().items[1].model_copy(update={"unit_price": 220.0, "line_total": 2200.0})],
        subtotal=9200.0, tax=920.0, grand_total=10120.0,
    )
    app, cfg = run(settings, contract, bad_text, [bad])
    snap = app.get_state(cfg)
    assert snap.next == ("human_review",) # paused
    assert snap.values["recommendation"] == "MANUAL_REVIEW"
    assert {f["code"] for f in snap.values["findings"]} == {"PAYMENT_TERMS_MISMATCH", "RATE_CEILING_BREACH"}

    app.invoke(Command(resume={"decision": "REJECT", "reviewer": "j.doe"}), cfg)
    final = app.get_state(cfg).values
    assert final["status"] == "RETURNED_TO_VENDOR"
    rec = log_lines(settings)[-1]
    assert rec["human_decision"]["reviewer"] == "j.doe" and rec["final_status"] == "RETURNED_TO_VENDOR"


def test_extraction_failure_escalates_to_human(settings, contract, clean_text):
    app, cfg = run(settings, contract, clean_text, [ConnectionError("down"), ConnectionError("down")])
    snap = app.get_state(cfg)
    assert snap.next == ("human_review",)
    assert snap.values["findings"][0]["code"] == "EXTRACTION_FAILED"
    assert snap.values["risk_score"] == 1.0


def test_duplicate_is_recommended_for_rejection(settings, contract, clean_text):
    app, cfg = run(settings, contract, clean_text, [make_extraction()], known=["INV-2026-9040"])
    assert app.get_state(cfg).values["recommendation"] == "REJECT"


def test_invalid_reviewer_decision_is_refused(settings, contract, clean_text):
    import pytest
    app, cfg = run(settings, contract, clean_text, [make_extraction()], known=["INV-2026-9040"])
    with pytest.raises(ValueError):
        app.invoke(Command(resume={"decision": "MAYBE"}), cfg)

# invoice layouts
import random

sys.path.insert(0, str(ROOT / "scripts"))
from evaluations import OracleLLM, build_clean
from layouts import LAYOUTS, render


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_layout_is_grounded_and_a_clean_invoice_passes(layout, contract, settings):
    rng = random.Random(3)
    truth = build_clean(rng, contract, 1)
    llm = OracleLLM()
    llm.truth = truth
    res = extract_invoice(render(truth, layout, rng), [("oracle", llm)], settings)
    assert res.confidence == 1.0, res.ungrounded
    assert run_rules(res.invoice, contract, [], settings) == []


def test_payment_terms_phrasings():
    assert parse_payment_terms_days("45 days net") == 45
    assert parse_payment_terms_days("payable within 30 days") == 30
