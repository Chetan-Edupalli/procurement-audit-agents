"""Run the agent over every invoice in a folder, like an AP inbox, and summarise the outcomes.
Non-interactive: invoices needing a human stay queued as AWAITING REVIEW.

    python scripts/batch_run.py                     # data/invoices/batch
    python scripts/batch_run.py path\\to\\folder --out results.csv
"""
import argparse
import csv
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from audit.agents import load_document
from audit.workflow import build_app, load_contract 


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", type=Path, default=ROOT / "data/invoices/batch")
    ap.add_argument("--contract", type=Path, default=ROOT / "data/contracts/apex_msa.json")
    ap.add_argument("--out", default="batch_results.csv")
    args = ap.parse_args()

    app, _ = build_app()
    contract = load_contract(args.contract)
    known_ids: list[str] = []      # running ledger: a repeated invoice id is flagged as a duplicate
    rows = []

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {".txt", ".pdf"})
    print(f"Processing {len(files)} invoices from {args.folder}\n")
    for path in files:
        t0 = time.perf_counter()
        row = {"file": path.name, "confidence": "", "recommendation": "", "risk": "", "at_risk": "",
               "findings": "", "outcome": ""}
        try:
            text = load_document(path)
        except Exception as exc:
            row["outcome"] = f"UNREADABLE ({exc})"
            rows.append(row)
            print(f"{path.name:<34} UNREADABLE")
            continue

        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        app.invoke({"raw_text": text, "contract": contract, "known_invoice_ids": list(known_ids), "trace": []}, cfg)
        snap = app.get_state(cfg)
        v = snap.values
        if v.get("invoice"):
            known_ids.append(v["invoice"]["invoice_id"])
        row.update(
            confidence=v["extraction_confidence"], recommendation=v["recommendation"], risk=v["risk_score"],
            at_risk=v["amount_at_risk"], findings=" ".join(f["code"] for f in v["findings"]),
            outcome="AWAITING REVIEW" if snap.next else "AUTO-POSTED",
        )
        rows.append(row)
        print(f"{path.name:<34} {row['outcome']:<16} {row['recommendation']:<14} conf={row['confidence']:<5} "
              f"risk={row['risk']:<5} {time.perf_counter() - t0:4.1f}s  {row['findings']}")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    posted = sum(r["outcome"] == "AUTO-POSTED" for r in rows)
    waiting = sum(r["outcome"] == "AWAITING REVIEW" for r in rows)
    print(f"\n{len(rows)} invoices: {posted} auto-posted, {waiting} awaiting review, "
          f"{len(rows) - posted - waiting} unreadable. Details: {args.out}")


if __name__ == "__main__":
    main()
