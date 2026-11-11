"""CLI demo.  python scripts/run_demo.py data/invoices/inv_violations.txt"""
import argparse
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from audit.agents import load_document
from audit.workflow import build_app, load_contract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("invoice", type=Path)
    ap.add_argument("--contract", type=Path, default=ROOT / "data/contracts/apex_msa.json")
    ap.add_argument("--known-ids", nargs="*", default=[], help="previously processed invoice ids")
    ap.add_argument("--reviewer", default="controller")
    ap.add_argument("--decision", choices=["APPROVE", "REJECT"], help="non-interactive reviewer decision")
    args = ap.parse_args()

    app, _ = build_app()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    app.invoke({"raw_text": load_document(args.invoice), "contract": load_contract(args.contract),
                "known_invoice_ids": args.known_ids, "trace": []}, cfg)

    snap = app.get_state(cfg)
    if snap.next: # paused at human_review
        v = snap.values
        print(f"\n=== ESCALATED: {v['recommendation']} | risk {v['risk_score']} | at risk {v['amount_at_risk']} ===")
        for f in v["findings"]:
            print(f" [{f['severity']}] {f['code']}: {f['message']}")
        print("\n--- Draft memo ---\n" + v["memo"] + "\n\n--- Draft vendor email ---\n" + v["vendor_email"])
        decision = args.decision or input("\nDecision [APPROVE/REJECT]: ").strip().upper()
        app.invoke(Command(resume={"decision": decision, "reviewer": args.reviewer}), cfg)

    final = app.get_state(cfg).values
    print(f"\nFINAL STATUS: {final['status']}")
    print("TRACE:", *final["trace"], sep="\n  ")


if __name__ == "__main__":
    main()
