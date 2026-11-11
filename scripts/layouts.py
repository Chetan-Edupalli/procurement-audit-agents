"""Invoice layouts for evaluation and demos. Each renderer turns a ground-truth Invoice into text
the way a different supplier (or a scanner) might print it.

    python scripts/layouts.py            # writes sample invoices to data/invoices/batch/
"""
import random
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from audit.schemas import Invoice


# formatting helpers
def us(d: Decimal) -> str:
    return f"{d:,.2f}"


def eu(d: Decimal) -> str:
    return us(d).replace(",", "X").replace(".", ",").replace("X", ".")


def qty(d: Decimal) -> str:
    return f"{d.normalize():f}"


def days_phrase(terms: str, template: str) -> str:
    """'Net 45' -> template.format(45); anything else is printed as-is."""
    m = re.search(r"net\s*(\d+)", terms, re.I)
    return template.format(m.group(1)) if m else terms


# layouts
def classic(inv: Invoice, rng: random.Random) -> str:
    lines = [f"INVOICE #{inv.invoice_id}", f"Vendor: {inv.vendor_name}", f"Payment Terms: {inv.payment_terms}", "",
             "Line Items:"]
    for n, i in enumerate(inv.items, 1):
        lines.append(f"{n}. {i.description} - {qty(i.quantity)} hrs @ ${us(i.unit_price)}/hr = ${us(i.line_total)}")
    return "\n".join(lines + ["", f"Subtotal: ${us(inv.subtotal)}", f"Tax: ${us(inv.tax)}",
                              f"Grand Total: ${us(inv.grand_total)}"])


def table(inv: Invoice, rng: random.Random) -> str:
    rate = inv.tax / inv.subtotal * 100 if inv.subtotal else Decimal(0)
    rows = [f"{n:>2} | {i.description:<42} | {qty(i.quantity):>5} | {us(i.unit_price):>10} | {us(i.line_total):>12}"
            for n, i in enumerate(inv.items, 1)]
    return "\n".join([
        f"INVOICE NO: {inv.invoice_id}        DATE: 2026-03-14",
        f"BILL FROM:  {inv.vendor_name}", f"TERMS:      {inv.payment_terms}", "",
        " # | Description                                | Qty   |       Rate |       Amount", "-" * 82, *rows, "-" * 82,
        f"{'Subtotal':>60} {us(inv.subtotal):>14}", f"{f'Tax ({rate:.0f}%)':>60} {us(inv.tax):>14}",
        f"{'TOTAL DUE (USD)':>60} {us(inv.grand_total):>14}"])


def european(inv: Invoice, rng: random.Random) -> str:
    lines = [f"Invoice No.: {inv.invoice_id}", f"Supplier: {inv.vendor_name}",
             f"Payment: {days_phrase(inv.payment_terms, '{} days net')}", "Currency: USD", ""]
    for n, i in enumerate(inv.items, 1):
        lines.append(f"Pos {n}: {i.description}; {qty(i.quantity)} h x {eu(i.unit_price)} = {eu(i.line_total)}")
    return "\n".join(lines + ["", f"Net amount: {eu(inv.subtotal)}", f"VAT: {eu(inv.tax)}",
                              f"Amount payable: {eu(inv.grand_total)}"])


def prose(inv: Invoice, rng: random.Random) -> str:
    bill = "\n".join(f" - {qty(i.quantity)} hours of {i.description} at ${us(i.unit_price)} per hour, "
                     f"which comes to ${us(i.line_total)}" for i in inv.items)
    return (f"Subject: Invoice {inv.invoice_id}\n\nHi Accounts Payable team,\n\nPlease find our invoice "
            f"{inv.invoice_id} for services delivered this month. We billed:\n{bill}\n\n"
            f"The subtotal is ${us(inv.subtotal)} and tax is ${us(inv.tax)}, so the total due is "
            f"${us(inv.grand_total)}. Payment terms: {inv.payment_terms}.\n\nKind regards,\nBilling Team\n"
            f"{inv.vendor_name}")


def totals_first(inv: Invoice, rng: random.Random) -> str:
    detail = []
    for n, i in enumerate(inv.items, 1):
        detail += [f"Item {n}: {i.description}",
                   f"  Hours: {qty(i.quantity)}   Rate: ${us(i.unit_price)}   Amount: ${us(i.line_total)}"]
    return "\n".join([
        "INVOICE SUMMARY", f"Reference: {inv.invoice_id}", f"Billed by: {inv.vendor_name}",
        f"Amount due: ${us(inv.grand_total)} USD", f"  of which tax: ${us(inv.tax)}",
        f"  net amount: ${us(inv.subtotal)}",
        f"Terms: {days_phrase(inv.payment_terms, 'payable within {} days')}", "", "DETAIL", *detail])


def noisy_scan(inv: Invoice, rng: random.Random) -> str:
    """Classic layout plus scanner/OCR-style mess and distractor numbers (PO, phone, bank account)."""
    body = []
    for line in classic(inv, rng).splitlines():
        body.append(re.sub(" ", lambda _m: "   " if rng.random() < 0.25 else " ", line))
        if rng.random() < 0.25:
            body.append("")
    head = ["*** SCANNED COPY - PAGE 1 OF 1 ***",
            f"PO: PO-{rng.randint(10000, 99999)}    Tel: +1 555 {rng.randint(100, 999)} {rng.randint(1000, 9999)}"]
    foot = ["", f"Remit to: First National Bank   Acct {rng.randint(10**9, 10**10 - 1)}",
            "Questions? accounts@vendor.example", "Thank you for your business!"]
    return "\n".join(head + body + foot)


LAYOUTS = {f.__name__: f for f in (classic, table, european, prose, totals_first, noisy_scan)}


def render(inv: Invoice, layout: str = "classic", rng: random.Random | None = None) -> str:
    return LAYOUTS[layout](inv, rng or random.Random(0))


# sample files
def write_samples(out_dir: Path) -> None:
    from evaluations import build_clean, inject

    from audit.schemas import Contract
    contract = Contract.model_validate_json((ROOT / "data/contracts/apex_msa.json").read_text())
    rng = random.Random(11)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = 100
    classic_clean_text = ""
    for name in LAYOUTS:
        for kind in ("clean", "faulty"):
            truth = build_clean(rng, contract, seq)
            seq += 1
            if kind == "faulty":
                truth = inject(inject(truth, "rate", rng, contract), "terms", rng, contract)
            text = render(truth, name, rng)
            (out_dir / f"inv_{name}_{kind}.txt").write_text(text, encoding="utf-8")
            if (name, kind) == ("classic", "clean"):
                classic_clean_text = text

    # hand-made edge cases: the system must fail SAFE (escalate to a human), never auto-approve
    clean = classic_clean_text
    edge = {
        "missing_total": "\n".join(l for l in clean.splitlines() if not l.startswith("Grand Total")),
        "ocr_garbled": clean.replace("7,0", "7,O").replace("10 hrs", "l0 hrs").replace("$1", "$l"),
        "not_an_invoice": "Hi all, lunch is at 12:30 on Friday. Table booked for 8 under Sam. Cheers!",
        "duplicate": clean + "\n(resubmitted)",
    }
    for name, text in edge.items():
        (out_dir / f"inv_edge_{name}.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {len(LAYOUTS) * 2 + len(edge)} invoices to {out_dir}")


if __name__ == "__main__":
    write_samples(ROOT / "data/invoices/batch")
