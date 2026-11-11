"""Grounding: every number an LLM extracts must appear verbatim in the source document.
The share that does is the extraction confidence score."""
import re
from decimal import Decimal

from .schemas import Invoice

def _variants(d: Decimal) -> set[str]:
    two = f"{d:.2f}"
    return {two, two.rstrip("0").rstrip("."), format(d, "f")}


def check_grounding(invoice: Invoice, raw_text: str) -> tuple[float, list[str]]:
    """Return (confidence, ungrounded_fields). Confidence = grounded numbers / checked numbers."""
    stripped = re.sub(r"[$€£]|\b(?:USD|EUR|GBP)\b", "", raw_text)
    texts = (
        stripped.replace(",", ""),
        stripped.replace(".", "").replace(",", "."),
    )
    checks: list[tuple[str, Decimal]] = [
        ("subtotal", invoice.subtotal),
        ("tax", invoice.tax),
        ("grand_total", invoice.grand_total),
    ]
    for n, item in enumerate(invoice.items, start=1):
        checks += [
            (f"item{n}.quantity", item.quantity),
            (f"item{n}.unit_price", item.unit_price),
            (f"item{n}.line_total", item.line_total),
        ]

    ungrounded = []
    for name, value in checks:
        found = any(re.search(rf"(?<![\d.]){re.escape(v)}(?!\d)", t) for t in texts for v in _variants(value))
        if not found:
            ungrounded.append(f"{name}={value}")
    total = len(checks) + 1  # +1 for the invoice id
    if not invoice.invoice_id or invoice.invoice_id not in raw_text:
        ungrounded.append(f"invoice_id={invoice.invoice_id!r}")
    confidence = 1 - len(ungrounded) / total
    return round(confidence, 3), ungrounded
