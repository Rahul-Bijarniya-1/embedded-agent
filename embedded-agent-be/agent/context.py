"""Page contexts: the displayed data of a page plus facts computed exactly in Python.

Only fields the page renders enter a context. Aggregates are computed over every row the
page sent, even when the rows themselves are truncated for the prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal

from .money import format_money
from .schemas import LoanData, TransactionRow

Page = Literal["transactions", "loan", "section"]


@dataclass
class PageContext:
    page: Page
    page_title: str
    facts: dict[str, Any]
    rows: list[dict[str, Any]]
    rows_total: int
    truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        payload = {
            "facts": self.facts,
            "rows": self.rows,
            "rows_total": self.rows_total,
            "rows_included": len(self.rows),
            "truncated": self.truncated,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def metadata_text(self) -> str:
        return json.dumps(self.metadata, ensure_ascii=False, default=str)


def _metadata(today: date, locale: str) -> dict[str, Any]:
    return {"today": today.isoformat(), "locale": locale}


def _num(value: float) -> int | float:
    value = float(value)
    return int(value) if value.is_integer() else round(value, 2)


def _with_money(fields: dict[str, Any], money_keys: list[str], locale: str) -> dict[str, Any]:
    """Add a `<key>_display` string next to each money key (keeps key order readable)."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        out[key] = value
        if key in money_keys:
            out[f"{key}_display"] = format_money(value, locale)
    return out


# --- Transactions ----------------------------------------------------------------------

def _grouped(rows: list[TransactionRow], key_fn: Callable[[TransactionRow], str], key_name: str, locale: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = buckets.setdefault(key_fn(row), {"debit_total": 0.0, "credit_total": 0.0, "count": 0})
        bucket["debit_total" if row.type == "debit" else "credit_total"] += row.amount
        bucket["count"] += 1
    return [
        _with_money(
            {key_name: key, "debit_total": _num(b["debit_total"]), "credit_total": _num(b["credit_total"]), "count": int(b["count"])},
            ["debit_total", "credit_total"],
            locale,
        )
        for key, b in buckets.items()
    ]


def _largest(rows: list[TransactionRow], locale: str) -> dict[str, Any] | None:
    if not rows:
        return None
    row = max(rows, key=lambda r: r.amount)
    return _with_money(
        {"date": row.date, "merchant": row.merchant, "category": row.category, "amount": _num(row.amount)},
        ["amount"],
        locale,
    )


def build_transactions_context(
    rows: list[TransactionRow], *, locale: str, today: date, max_rows: int = 500
) -> PageContext:
    debits = [r for r in rows if r.type == "debit"]
    credits = [r for r in rows if r.type == "credit"]
    total_debit = sum(r.amount for r in debits)
    total_credit = sum(r.amount for r in credits)
    by_spend = lambda b: (-b["debit_total"], -b["credit_total"])  # noqa: E731

    facts: dict[str, Any] = {
        "count": len(rows),
        "date_range": {"from": min(r.date for r in rows), "to": max(r.date for r in rows)},
        **_with_money(
            {
                "total_debit": _num(total_debit),
                "total_credit": _num(total_credit),
                "net": _num(total_credit - total_debit),
                "average_debit": round(total_debit / len(debits), 2) if debits else 0,
            },
            ["total_debit", "total_credit", "net", "average_debit"],
            locale,
        ),
        "largest_debit": _largest(debits, locale),
        "largest_credit": _largest(credits, locale),
        "by_category": sorted(_grouped(rows, lambda r: r.category, "category", locale), key=by_spend),
        "by_merchant": sorted(_grouped(rows, lambda r: r.merchant, "merchant", locale), key=by_spend),
        "by_month": sorted(_grouped(rows, lambda r: r.date[:7], "month", locale), key=lambda b: b["month"]),
    }
    out_rows = [
        _with_money(
            {"date": r.date, "category": r.category, "merchant": r.merchant, "type": r.type, "amount": _num(r.amount)},
            ["amount"],
            locale,
        )
        for r in rows[:max_rows]
    ]
    return PageContext(
        page="transactions",
        page_title="Transactions (6 months)",
        facts=facts,
        rows=out_rows,
        rows_total=len(rows),
        truncated=len(rows) > max_rows,
        metadata=_metadata(today, locale),
    )


# --- Loan details ----------------------------------------------------------------------

_LOAN_MONEY = ["outstanding", "original_amount", "emi", "total_paid", "total_interest_paid", "late_payment_fee"]


def build_loan_context(
    loan: LoanData, *, locale: str, today: date, max_history_rows: int = 60
) -> PageContext:
    scalars: dict[str, Any] = {
        "name": loan.name,
        "type": loan.type,
        "status": loan.status,
        "lender": loan.lender,
        "account_number": loan.account_number,
        "outstanding": _num(loan.outstanding),
        "original_amount": _num(loan.original_amount),
        "emi": _num(loan.emi),
        "interest_rate": loan.interest_rate,
        "total_paid": None if loan.total_paid is None else _num(loan.total_paid),
        "total_interest_paid": None if loan.total_interest_paid is None else _num(loan.total_interest_paid),
        "disbursement_date": loan.disbursement_date,
        "maturity_date": loan.maturity_date,
    }
    if loan.status == "Active":
        scalars["next_due_date"] = loan.next_due_date
        scalars["tenure_remaining"] = loan.tenure_remaining
    else:
        scalars["closed_date"] = loan.closed_date
    scalars["prepayment_charges"] = loan.prepayment_charges
    scalars["late_payment_fee"] = None if loan.late_payment_fee is None else _num(loan.late_payment_fee)
    facts = _with_money(scalars, _LOAN_MONEY, locale)

    if loan.vehicle_details:
        facts["vehicle_details"] = {k: v for k, v in loan.vehicle_details.model_dump().items() if v is not None}
    if loan.property_details:
        details = loan.property_details.model_dump()
        if details.get("property_value") is not None:
            details["property_value"] = _num(details["property_value"])
        facts["property_details"] = _with_money(details, ["property_value"], locale)

    history = loan.payment_history
    status_of = lambda r: r.status.strip().lower()  # noqa: E731
    facts["shown_history"] = {
        "months": len(history),
        "paid": sum(1 for r in history if status_of(r) == "paid"),
        "late": sum(1 for r in history if status_of(r) == "late"),
        "missed": sum(1 for r in history if status_of(r) == "missed"),
        "pending": sum(1 for r in history if status_of(r) == "pending"),
        **_with_money(
            {
                "amount_paid_in_shown_history": _num(sum(r.amount for r in history)),
                "principal_in_shown_history": _num(sum(r.principal for r in history)),
                "interest_in_shown_history": _num(sum(r.interest for r in history)),
            },
            ["amount_paid_in_shown_history", "principal_in_shown_history", "interest_in_shown_history"],
            locale,
        ),
        "late_months": [r.month_year for r in history if status_of(r) == "late"],
        "missed_months": [r.month_year for r in history if status_of(r) == "missed"],
        "pending_months": [r.month_year for r in history if status_of(r) == "pending"],
    }

    rows = [
        {
            "month_year": r.month_year,
            "due_date": r.due_date,
            "paid_date": r.paid_date,
            "amount": _num(r.amount),
            "amount_display": format_money(r.amount, locale),
            "principal": _num(r.principal),
            "interest": _num(r.interest),
            "status": r.status,
        }
        for r in history[:max_history_rows]
    ]
    return PageContext(
        page="loan",
        page_title=f"{loan.name} - Detailed View",
        facts=facts,
        rows=rows,
        rows_total=len(history),
        truncated=len(history) > max_history_rows,
        metadata=_metadata(today, locale),
    )


# --- Explain (any section of any page) -------------------------------------------------

def build_section_context(
    page: str, section: str, section_text: str, data: Any, *, locale: str, today: date
) -> PageContext:
    return PageContext(
        page="section",
        page_title=f"{page} › {section}",
        facts={"page": page, "section": section, "section_text": section_text, "data": data},
        rows=[],
        rows_total=0,
        truncated=False,
        metadata=_metadata(today, locale),
    )
