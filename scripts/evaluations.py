"""Evaluation harness: how well does the pipeline catch seeded errors, across invoice layouts?

Generates synthetic invoices with known injected faults, runs them through extraction + rules,
and compares against the findings produced from the ground-truth invoice.

  python scripts/evaluations.py --n 50 --mock    # oracle extractor: validates harness + rules only
  python scripts/evaluations.py --n 50           # real LLM extraction via Ollama (the number to report)

Headline metric: FALSE-APPROVAL RATE - faulty invoices the system would have paid automatically.
"""
import argparse
import json
import random
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from audit.agents import extract_invoice
from audit.rules import recommend, run_rules
from audit.schemas import Contract, Invoice, LineItem, Settings
from layouts import LAYOUTS, render

D = Decimal
FAULTS = ["rate", "terms", "subtotal", "tax", "duplicate", "uncontracted", "line_total"]


def build_clean(rng: random.Random, contract: Contract, n: int) -> Invoice:
    items = []
    for entry in rng.sample(contract.rate_card, k=rng.randint(1, len(contract.rate_card))):
        qty = D(rng.randint(5, 80))
        price = (entry.max_unit_price * D(str(rng.uniform(0.8, 1.0)))).quantize(D("0.01"))
        items.append(LineItem(description=f"Enterprise {entry.service} service", quantity=qty,
                              unit_price=price, line_total=qty * price))
    sub = sum(i.line_total for i in items)
    tax = (sub * contract.tax_rate).quantize(D("0.01"))
    return Invoice(vendor_name=contract.vendor_name, invoice_id=f"INV-EVAL-{n:04d}",
                   payment_terms=f"Net {contract.payment_terms_days}", items=items,
                   subtotal=sub, tax=tax, grand_total=sub + tax)


def inject(inv: Invoice, fault: str, rng: random.Random, contract: Contract) -> Invoice:
    inv = inv.model_copy(deep=True)

    def reprice():
        inv.subtotal = sum(i.line_total for i in inv.items)
        inv.tax = (inv.subtotal * contract.tax_rate).quantize(D("0.01"))
        inv.grand_total = inv.subtotal + inv.tax

    if fault == "rate":
        it = inv.items[0]
        ceiling = next(e.max_unit_price for e in contract.rate_card if e.service in it.description)
        it.unit_price = (ceiling * D(str(rng.uniform(1.05, 1.3)))).quantize(D("0.01"))
        it.line_total = it.quantity * it.unit_price
        reprice()
    elif fault == "terms":
        inv.payment_terms = rng.choice(["Net 15", "Net 30", "Due on receipt"])
    elif fault == "subtotal":
        inv.subtotal += D(rng.randint(100, 900))
        inv.grand_total = inv.subtotal + inv.tax
    elif fault == "tax":
        inv.tax = (inv.subtotal * D("0.13")).quantize(D("0.01"))
        inv.grand_total = inv.subtotal + inv.tax
    elif fault == "uncontracted":
        inv.items.append(LineItem(description="Premium Concierge Support", quantity=D(5),
                                  unit_price=D("300.00"), line_total=D("1500.00")))
        reprice()
    elif fault == "line_total":
        inv.items[0].line_total += D(rng.randint(50, 400))
        inv.subtotal = sum(i.line_total for i in inv.items)
        inv.grand_total = inv.subtotal + inv.tax
    return inv  # "duplicate" is injected via known_invoice_ids, not by changing the document


class OracleLLM:
    """--mock mode: returns the ground truth, so only harness and rules are exercised."""

    def __init__(self):
        self.truth = None

    def with_structured_output(self, _schema):
        from audit.schemas import InvoiceExtraction, LineItemExtraction
        t = self.truth
        return type("S", (), {"invoke": lambda _s, _m: InvoiceExtraction(
            vendor_name=t.vendor_name, invoice_id=t.invoice_id, payment_terms=t.payment_terms,
            items=[LineItemExtraction(description=i.description, quantity=float(i.quantity),
                                      unit_price=float(i.unit_price), line_total=float(i.line_total)) for i in t.items],
            subtotal=float(t.subtotal), tax=float(t.tax), grand_total=float(t.grand_total))})()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--layouts", default="all", help="comma list of layouts, or 'all'")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    rng, settings = random.Random(args.seed), Settings(audit_log_path="/dev/null")
    contract = Contract.model_validate_json((ROOT / "data/contracts/apex_msa.json").read_text())
    if args.mock:
        llm = OracleLLM()
        llms = [("oracle", llm)]
    else:
        from langchain_ollama import ChatOllama
        llms = [(settings.primary_model, ChatOllama(model=settings.primary_model, temperature=0, base_url=settings.ollama_url))]

    names = list(LAYOUTS) if args.layouts == "all" else args.layouts.split(",")
    per = {n: dict(n=0, faulty=0, false_approve=0, clean=0, false_escalate=0, tp=0, fp=0, fn=0, failures=0)
           for n in names}
    latencies = []
    for n in range(args.n):
        truth = build_clean(rng, contract, n)
        injected = rng.sample(FAULTS, k=rng.choice([0, 0, 1, 1, 2]))
        known = []
        for f in injected:
            truth = inject(truth, f, rng, contract)
        if "duplicate" in injected:
            known = [truth.invoice_id]
        expected = {f.code for f in run_rules(truth, contract, known, settings)}
        layout = rng.choice(names)
        p = per[layout]
        p["n"] += 1

        if args.mock:
            llm.truth = truth
        t0 = time.perf_counter()
        res = extract_invoice(render(truth, layout, rng), llms, settings)
        latencies.append(time.perf_counter() - t0)

        if res.invoice is None:
            p["failures"] += 1
            found, rec = {"EXTRACTION_FAILED"}, "MANUAL_REVIEW"
        else:
            fs = run_rules(res.invoice, contract, known, settings)
            found = {f.code for f in fs}
            rec = recommend(fs, res.confidence, settings)
        p["tp"] += len(expected & found)
        p["fp"] += len(found - expected - {"EXTRACTION_FAILED"})
        p["fn"] += len(expected - found)
        if expected:
            p["faulty"] += 1
            p["false_approve"] += rec == "APPROVE"
        else:
            p["clean"] += 1
            p["false_escalate"] += rec != "APPROVE"

    def ratio(a, b):
        return round(a / b, 3) if b else None

    tot = {k: sum(p[k] for p in per.values()) for k in next(iter(per.values()))}
    out = {
        "n": args.n, "mode": "mock-oracle" if args.mock else llms[0][0],
        "faulty_invoices": tot["faulty"], "clean_invoices": tot["clean"],
        "finding_precision": ratio(tot["tp"], tot["tp"] + tot["fp"]),
        "finding_recall": ratio(tot["tp"], tot["tp"] + tot["fn"]),
        "false_approval_rate": ratio(tot["false_approve"], tot["faulty"]),
        "false_escalation_rate": ratio(tot["false_escalate"], tot["clean"]),
        "extraction_failures": tot["failures"],
        "mean_latency_s": round(sum(latencies) / len(latencies), 2),
        "by_layout": {
            n: {"n": p["n"], "recall": ratio(p["tp"], p["tp"] + p["fn"]),
                "false_approval_rate": ratio(p["false_approve"], p["faulty"]),
                "false_escalation_rate": ratio(p["false_escalate"], p["clean"]),
                "extraction_failures": p["failures"]}
            for n, p in per.items()
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
