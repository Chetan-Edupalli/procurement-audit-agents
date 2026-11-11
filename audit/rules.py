"""Guardrails & controls: deterministic compliance rules, risk scoring and routing.

Design principle: the LLM reads documents and writes prose; *code* makes the control decisions.
Every rule is a pure function an auditor can read and a unit test can pin down.
"""
import re
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence

from .schemas import Contract, Finding, Invoice, Recommendation, Settings, Severity

_ZERO = Decimal("0")
_SUFFIXES = {"ltd", "limited", "inc", "llc", "corp", "corporation", "co", "plc", "gmbh"}


def parse_payment_terms_days(terms: str) -> Optional[int]:
    t = terms.lower()
    if m := re.search(r"net\s*(\d+)", t):
        return int(m.group(1))
    if m := re.search(r"(\d+)\s*(?:calendar\s*)?days", t):  # "45 days net", "within 30 days"
        return int(m.group(1))
    if "receipt" in t:
        return 0
    return None


def _norm_vendor(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", name.lower()).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def run_rules(
    inv: Invoice,
    contract: Contract,
    known_invoice_ids: Iterable[str],
    settings: Settings,
) -> list[Finding]:
    tol = settings.amount_tolerance
    out: list[Finding] = []

    def add(code, sev, msg, at_risk=_ZERO):
        out.append(Finding(code=code, severity=sev, message=msg, amount_at_risk=max(at_risk, _ZERO)))

    # 1. Identity & duplicates
    sim = SequenceMatcher(None, _norm_vendor(inv.vendor_name), _norm_vendor(contract.vendor_name)).ratio()
    if sim < settings.vendor_name_similarity:
        add("VENDOR_MISMATCH", Severity.HIGH,
            f"Invoice vendor '{inv.vendor_name}' does not match contract counterparty '{contract.vendor_name}'.",
            inv.grand_total)
    if inv.invoice_id in set(known_invoice_ids):
        add("DUPLICATE_INVOICE", Severity.HIGH,
            f"Invoice {inv.invoice_id} was already processed for this vendor.", inv.grand_total)
    if inv.currency != contract.currency:
        add("CURRENCY_MISMATCH", Severity.HIGH,
            f"Invoice currency {inv.currency} differs from contract currency {contract.currency}.", inv.grand_total)

    # 2. Commercial terms
    days = parse_payment_terms_days(inv.payment_terms)
    if days is None:
        add("PAYMENT_TERMS_UNPARSEABLE", Severity.LOW, f"Could not interpret payment terms '{inv.payment_terms}'.")
    elif days != contract.payment_terms_days:
        shorter = days < contract.payment_terms_days
        add("PAYMENT_TERMS_MISMATCH", Severity.MEDIUM if shorter else Severity.LOW,
            f"Invoice terms Net {days}; contract requires Net {contract.payment_terms_days}"
            + (" (accelerates cash outflow)." if shorter else "."))

    # 3. Rate card, line by line
    for n, item in enumerate(inv.items, start=1):
        entry = next(
            (e for e in contract.rate_card if any(k.lower() in item.description.lower() for k in e.keywords)),
            None,
        )
        if entry is None:
            add("UNCONTRACTED_ITEM", Severity.MEDIUM,
                f"Line {n} '{item.description}' is not covered by the contract rate card.", item.line_total)
        elif item.unit_price > entry.max_unit_price:
            over = (item.unit_price - entry.max_unit_price) * item.quantity
            add("RATE_CEILING_BREACH", Severity.HIGH,
                f"Line {n} '{item.description}': {item.unit_price}/unit exceeds contracted ceiling "
                f"{entry.max_unit_price}/unit (overbilled {over:.2f}).", over)
        if abs(item.quantity * item.unit_price - item.line_total) > tol:
            diff = item.line_total - item.quantity * item.unit_price
            add("LINE_TOTAL_MISMATCH", Severity.MEDIUM,
                f"Line {n}: {item.quantity} x {item.unit_price} = {item.quantity * item.unit_price:.2f}, "
                f"but invoice states {item.line_total}.", diff)

    # 4. Arithmetic & tax
    line_sum = sum((i.line_total for i in inv.items), _ZERO)
    if abs(line_sum - inv.subtotal) > tol:
        add("SUBTOTAL_MISMATCH", Severity.HIGH,
            f"Line items sum to {line_sum:.2f}, but subtotal states {inv.subtotal:.2f}.", inv.subtotal - line_sum)
    if abs(inv.subtotal + inv.tax - inv.grand_total) > tol:
        add("GRAND_TOTAL_MISMATCH", Severity.HIGH,
            f"Subtotal + tax = {inv.subtotal + inv.tax:.2f}, but grand total states {inv.grand_total:.2f}.",
            inv.grand_total - (inv.subtotal + inv.tax))
    if inv.subtotal > 0:
        rate = inv.tax / inv.subtotal
        if abs(rate - contract.tax_rate) > settings.tax_rate_tolerance:
            add("TAX_RATE_MISMATCH", Severity.MEDIUM,
                f"Effective tax rate {rate:.2%} differs from contract rate {contract.tax_rate:.2%}.",
                inv.tax - inv.subtotal * contract.tax_rate)
    return out


# scoring & routing
# Probability-style weights combined with noisy-OR: score = 1 - prod(1 - w_i).
# Bounded in [0, 1], monotonic in the number/severity of findings, no arbitrary cap.
WEIGHTS = {Severity.LOW: 0.10, Severity.MEDIUM: 0.25, Severity.HIGH: 0.50}

# Findings where the correct action is to return the invoice, not to fix it up.
REJECT_CODES = {"DUPLICATE_INVOICE", "VENDOR_MISMATCH"}


def risk_score(findings: Sequence[Finding]) -> float:
    p_ok = 1.0
    for f in findings:
        p_ok *= 1 - WEIGHTS[f.severity]
    return round(1 - p_ok, 2)


def amount_at_risk(findings: Sequence[Finding]) -> Decimal:
    return sum((f.amount_at_risk for f in findings), Decimal("0"))


def recommend(findings: Sequence[Finding], extraction_confidence: float, settings: Settings) -> Recommendation:
    """Only a clean, high-confidence invoice is auto-approved. Everything else goes to a human,
    with REJECT as a *recommendation* for the reviewer; the system never rejects on its own."""
    if any(f.code in REJECT_CODES for f in findings):
        return "REJECT"
    if findings or extraction_confidence < settings.min_extraction_confidence:
        return "MANUAL_REVIEW"
    return "APPROVE"
