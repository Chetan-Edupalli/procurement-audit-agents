"""Settings and data models (Pydantic). Money is Decimal, never float."""
import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Settings:
    primary_model: str = field(default_factory=lambda: os.getenv("AUDIT_PRIMARY_MODEL", "qwen2.5:7b"))
    fallback_model: str = field(default_factory=lambda: os.getenv("AUDIT_FALLBACK_MODEL", "llama3.1:8b"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    audit_log_path: str = field(default_factory=lambda: os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl"))

    max_attempts_per_model: int = 2
    # Below this share of extracted numbers found verbatim in the source text,
    # the invoice is never auto-approved.
    min_extraction_confidence: float = 0.90
    amount_tolerance: Decimal = Decimal("0.02")   # rounding tolerance in currency units
    tax_rate_tolerance: Decimal = Decimal("0.005")  # 0.5 percentage points
    vendor_name_similarity: float = 0.85


# Domain models: money is Decimal, never float.
class LineItem(BaseModel):
    description: str
    quantity: Decimal
    unit_price: Decimal
    line_total: Decimal


class Invoice(BaseModel):
    vendor_name: str
    invoice_id: str
    payment_terms: str
    currency: str = "USD"
    items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal
    tax: Decimal
    grand_total: Decimal


# LLM-facing schema: deliberately simple (floats) so small local models can fill it.
class LineItemExtraction(BaseModel):
    description: str = Field(description="Description of the goods or service")
    quantity: float = Field(description="Units billed, e.g. hours")
    unit_price: float = Field(description="Price per unit")
    line_total: float = Field(description="Total for this line as printed on the invoice")


class InvoiceExtraction(BaseModel):
    vendor_name: str
    invoice_id: str
    payment_terms: str = Field(description="Exactly as printed, e.g. 'Net 30'")
    currency: str = Field(default="USD", description="ISO currency code")
    items: list[LineItemExtraction]
    subtotal: float
    tax: float
    grand_total: float

    def to_domain(self) -> Invoice:
        def d(x: float) -> Decimal:
            return Decimal(str(x))  # str() avoids binary-float artefacts

        return Invoice(
            vendor_name=self.vendor_name.strip(),
            invoice_id=self.invoice_id.strip(),
            payment_terms=self.payment_terms.strip(),
            currency=self.currency.strip().upper(),
            items=[
                LineItem(
                    description=i.description.strip(),
                    quantity=d(i.quantity),
                    unit_price=d(i.unit_price),
                    line_total=d(i.line_total),
                )
                for i in self.items
            ],
            subtotal=d(self.subtotal),
            tax=d(self.tax),
            grand_total=d(self.grand_total),
        )


class RateEntry(BaseModel):
    service: str
    keywords: list[str]
    max_unit_price: Decimal


class Contract(BaseModel):
    """Machine-readable slice of a Master Services Agreement."""
    contract_id: str
    vendor_name: str
    currency: str = "USD"
    payment_terms_days: int
    tax_rate: Decimal
    rate_card: list[RateEntry]


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str
    amount_at_risk: Decimal = Decimal("0")


Recommendation = Literal["APPROVE", "MANUAL_REVIEW", "REJECT"]


class Drafts(BaseModel):
    memo: str = Field(description="Executive audit memo, max 200 words")
    vendor_email: str = Field(description="Polite email to the vendor requesting correction")


class ExtractionResult(BaseModel):
    invoice: Optional[Invoice] = None
    confidence: float = 0.0
    model_used: Optional[str] = None
    attempts: int = 0
    ungrounded: list[str] = []
    error: Optional[str] = None
