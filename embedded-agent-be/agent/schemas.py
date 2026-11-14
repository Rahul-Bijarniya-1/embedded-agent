"""Request and response models.

The field lists here are the whitelist of what each page displays: anything the page
sends that is not declared is dropped before it can reach a prompt.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

MAX_QUESTION_CHARS = 500
MAX_TRANSACTION_ROWS = 2000
MAX_SECTION_TEXT_CHARS = 4000
MAX_EXPLAIN_DATA_CHARS = 20_000
MAX_TTS_CHARS = 2000
DEFAULT_LOCALE = "en-US"

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOCALE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


def clean_text_input(value: object, *, max_chars: int, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = _CONTROL_CHARS.sub("", value).strip()
    if not text:
        raise ValueError(f"{label} must not be blank")
    if len(text) > max_chars:
        raise ValueError(f"{label} must be at most {max_chars} characters")
    return text


def clean_question(value: object) -> str:
    return clean_text_input(value, max_chars=MAX_QUESTION_CHARS, label="question")


def clean_tts_text(value: object) -> str:
    return clean_text_input(value, max_chars=MAX_TTS_CHARS, label="text")


def normalise_locale(value: object) -> str:
    text = str(value or "").strip()
    if len(text) > 16 or not _LOCALE.match(text):
        return DEFAULT_LOCALE
    return text


Question = Annotated[str, BeforeValidator(clean_question)]
Locale = Annotated[str, BeforeValidator(normalise_locale)]


class _Lenient(BaseModel):
    """Ignore unknown keys: the page may send more than it displays."""

    model_config = ConfigDict(extra="ignore")


# --- Transactions page -----------------------------------------------------------------

class TransactionRow(_Lenient):
    date: str
    category: str
    merchant: str
    type: Literal["debit", "credit"]
    amount: float = Field(ge=0)


Transactions = Annotated[list[TransactionRow], Field(min_length=1, max_length=MAX_TRANSACTION_ROWS)]


class TransactionsSummarizeRequest(_Lenient):
    transactions: Transactions
    locale: Locale = DEFAULT_LOCALE


class TransactionsAskRequest(_Lenient):
    question: Question
    transactions: Transactions
    locale: Locale = DEFAULT_LOCALE


# --- Loan details page -----------------------------------------------------------------

class PaymentRow(_Lenient):
    month_year: str
    due_date: str | None = None
    paid_date: str | None = None
    amount: float = 0
    principal: float = 0
    interest: float = 0
    status: str


class VehicleDetails(_Lenient):
    make: str | None = None
    model: str | None = None
    year: int | str | None = None
    registration: str | None = None


class PropertyDetails(_Lenient):
    address: str | None = None
    type: str | None = None
    property_value: float | None = None


class LoanData(_Lenient):
    name: str
    type: str
    status: str
    lender: str | None = None
    account_number: str | None = None
    outstanding: float
    original_amount: float
    emi: float
    interest_rate: float
    total_paid: float | None = None
    total_interest_paid: float | None = None
    disbursement_date: str | None = None
    maturity_date: str | None = None
    next_due_date: str | None = None
    closed_date: str | None = None
    tenure_remaining: str | None = None
    prepayment_charges: str | None = None
    late_payment_fee: float | None = None
    vehicle_details: VehicleDetails | None = None
    property_details: PropertyDetails | None = None
    payment_history: list[PaymentRow] = Field(default_factory=list)


class LoanSummarizeRequest(_Lenient):
    loan_id: str | None = None
    loan_data: LoanData
    locale: Locale = DEFAULT_LOCALE


class LoanAskRequest(_Lenient):
    loan_id: str | None = None
    question: Question
    loan_data: LoanData
    locale: Locale = DEFAULT_LOCALE


# --- Explain and speech ----------------------------------------------------------------

class ExplainRequest(_Lenient):
    page: str = Field(min_length=1, max_length=64)
    section: str = Field(min_length=1, max_length=64)
    section_text: str = Field("", max_length=MAX_SECTION_TEXT_CHARS)
    data: Any | None = None
    locale: Locale = DEFAULT_LOCALE
    refresh: bool = False  # the page's Refresh button: skip the cached wording and generate again

    @field_validator("data")
    @classmethod
    def _data_must_be_small(cls, value: Any) -> Any:
        if value is not None and len(json.dumps(value, default=str)) > MAX_EXPLAIN_DATA_CHARS:
            raise ValueError(f"data must serialise to at most {MAX_EXPLAIN_DATA_CHARS} characters")
        return value


class TtsRequest(_Lenient):
    text: Annotated[str, BeforeValidator(clean_tts_text)]
    locale: Locale = DEFAULT_LOCALE


# --- Responses -------------------------------------------------------------------------

Scope = Literal["in_page", "general_term", "out_of_page", "unknown"]


class Meta(BaseModel):
    model: str
    latency_ms: int
    cached: bool = False
    request_id: str = ""


class AnswerResponse(BaseModel):
    answer: str
    scope: Scope = "unknown"
    meta: Meta


class SummaryResponse(BaseModel):
    summary: str
    meta: Meta


class ExplainResponse(BaseModel):
    explanation: str
    meta: Meta


class ErrorResponse(BaseModel):
    error: str
    message: str
