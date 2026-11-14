import json
from datetime import date

from agent.context import (
    PageContext,
    build_loan_context,
    build_section_context,
    build_transactions_context,
)
from agent.schemas import LoanData, TransactionRow

TODAY = date(2026, 11, 13)

ROWS = [
    TransactionRow.model_validate(r)
    for r in [
        {"date": "2025-01-05", "category": "Food", "merchant": "Starbucks", "type": "debit", "amount": 300},
        {"date": "2025-01-20", "category": "Shopping", "merchant": "Amazon", "type": "debit", "amount": 1200},
        {"date": "2025-02-02", "category": "Food", "merchant": "Walmart", "type": "debit", "amount": 500},
        {"date": "2025-02-10", "category": "Travel", "merchant": "Uber", "type": "credit", "amount": 1000},
    ]
]

HISTORY = [
    {"month_year": "November 2026", "due_date": "2026-11-13", "paid_date": "2026-11-13", "amount": 4500, "principal": 3000, "interest": 1500, "status": "Paid"},
    {"month_year": "October 2026", "due_date": "2026-10-14", "paid_date": "2026-10-14", "amount": 4500, "principal": 3000, "interest": 1500, "status": "Paid"},
    {"month_year": "September 2026", "due_date": "2026-09-14", "paid_date": "2026-09-18", "amount": 4500, "principal": 3000, "interest": 1500, "status": "Late"},
    {"month_year": "August 2026", "due_date": "2026-08-15", "paid_date": None, "amount": 0, "principal": 0, "interest": 0, "status": "Missed"},
    {"month_year": "July 2026", "due_date": "2026-07-16", "paid_date": "2026-07-16", "amount": 4500, "principal": 3000, "interest": 1500, "status": "Paid"},
    {"month_year": "June 2026", "due_date": "2026-06-16", "paid_date": None, "amount": 0, "principal": 0, "interest": 0, "status": "Pending"},
]

ACTIVE = LoanData.model_validate({
    "id": "LOAN002", "name": "Car Loan", "type": "Auto", "outstanding": 120000, "original_amount": 500000,
    "interest_rate": 9.5, "emi": 8500, "next_due_date": "2026-11-23", "tenure_remaining": "36 months",
    "status": "Active", "disbursement_date": "2022-02-01", "maturity_date": "2027-02-01", "lender": "ICICI Bank",
    "account_number": "XXXX5678", "payment_history": HISTORY, "total_paid": 380000, "total_interest_paid": 95000,
    "prepayment_charges": "3% of outstanding", "late_payment_fee": 750,
    "vehicle_details": {"make": "Honda", "model": "City", "year": 2022, "registration": "MH01AB1234"},
})

CLOSED = LoanData.model_validate({
    "id": "LOAN004", "name": "Education Loan", "type": "Education", "outstanding": 0, "original_amount": 200000,
    "interest_rate": 10.5, "emi": 0, "next_due_date": None, "tenure_remaining": "0 months", "status": "Closed",
    "disbursement_date": "2020-07-01", "maturity_date": "2024-07-01", "closed_date": "2024-06-15", "lender": "SBI",
    "account_number": "XXXX3456", "payment_history": HISTORY, "total_paid": 200000, "total_interest_paid": 42000,
    "prepayment_charges": "Nil", "late_payment_fee": 0,
})

PROPERTY = LoanData.model_validate({
    **ACTIVE.model_dump(), "name": "Home Loan", "vehicle_details": None,
    "property_details": {"address": "123 Main Street, Mumbai", "property_value": 4000000, "type": "Apartment"},
})


def ctx(locale="en-US", **kwargs) -> PageContext:
    return build_transactions_context(ROWS, locale=locale, today=TODAY, **kwargs)


def without_display(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.endswith("_display")}


def test_transactions_page_identity():
    c = ctx()
    assert c.page == "transactions"
    assert c.page_title == "Transactions (6 months)"


def test_transactions_facts_totals():
    f = ctx().facts
    assert f["count"] == 4
    assert f["date_range"] == {"from": "2025-01-05", "to": "2025-02-10"}
    assert f["total_debit"] == 2000
    assert f["total_credit"] == 1000
    assert f["net"] == -1000


def test_transactions_by_category_sorted_by_spend():
    cats = ctx().facts["by_category"]
    assert [c["category"] for c in cats] == ["Shopping", "Food", "Travel"]
    assert without_display(cats[0]) == {"category": "Shopping", "debit_total": 1200, "credit_total": 0, "count": 1}
    assert without_display(cats[1]) == {"category": "Food", "debit_total": 800, "credit_total": 0, "count": 2}
    assert without_display(cats[2]) == {"category": "Travel", "debit_total": 0, "credit_total": 1000, "count": 1}
    assert cats[0]["debit_total_display"] == "₹1,200"


def test_transactions_by_merchant_sorted_by_spend():
    merchants = ctx().facts["by_merchant"]
    assert [m["merchant"] for m in merchants] == ["Amazon", "Walmart", "Starbucks", "Uber"]


def test_transactions_largest_and_average():
    f = ctx().facts
    assert f["largest_debit"] == {"date": "2025-01-20", "merchant": "Amazon", "category": "Shopping", "amount": 1200, "amount_display": "₹1,200"}
    assert f["largest_credit"]["merchant"] == "Uber"
    assert f["largest_credit"]["amount"] == 1000
    assert f["average_debit"] == 666.67
    assert f["average_debit_display"] == "₹666.67"


def test_transactions_largest_credit_is_null_without_credits():
    c = build_transactions_context(ROWS[:3], locale="en-US", today=TODAY)
    assert c.facts["largest_credit"] is None
    assert c.facts["total_credit"] == 0


def test_transactions_by_month():
    months = [without_display(m) for m in ctx().facts["by_month"]]
    assert months == [
        {"month": "2025-01", "debit_total": 1500, "credit_total": 0, "count": 2},
        {"month": "2025-02", "debit_total": 500, "credit_total": 1000, "count": 2},
    ]


def test_transactions_display_strings_follow_locale():
    assert ctx("en-US").facts["total_debit_display"] == "₹2,000"
    assert ctx("en-IN").facts["total_debit_display"] == "₹2,000"
    big = build_transactions_context(
        [TransactionRow(date="2025-03-01", category="Travel", merchant="Air India", type="debit", amount=120000)],
        locale="en-IN", today=TODAY,
    )
    assert big.facts["total_debit_display"] == "₹1,20,000"
    assert big.rows[0]["amount_display"] == "₹1,20,000"
    assert big.facts["net_display"] == "-₹1,20,000"


def test_transactions_rows_keep_only_displayed_columns_plus_display():
    row = ctx().rows[0]
    assert set(row) == {"date", "category", "merchant", "type", "amount", "amount_display"}
    assert row["amount"] == 300


def test_transactions_truncates_rows_but_facts_cover_all():
    rows = ROWS + ROWS[:3]
    c = build_transactions_context(rows, locale="en-US", today=TODAY, max_rows=5)
    assert len(c.rows) == 5
    assert c.rows_total == 7
    assert c.truncated is True
    assert c.facts["count"] == 7
    assert ctx().truncated is False


def test_metadata_has_today_and_locale():
    c = ctx("en-IN")
    assert c.metadata == {"today": "2026-11-13", "locale": "en-IN"}
    assert json.loads(c.metadata_text()) == c.metadata


def test_to_prompt_text_is_json():
    d = json.loads(ctx().to_prompt_text())
    assert set(d) == {"facts", "rows", "rows_total", "rows_included", "truncated"}
    assert d["rows_included"] == 4
    assert d["rows_total"] == 4
    assert d["facts"]["total_debit"] == 2000


def test_loan_context_active_fields():
    c = build_loan_context(ACTIVE, locale="en-US", today=TODAY)
    assert c.page == "loan"
    assert c.page_title == "Loan Details"
    f = c.facts
    assert f["status"] == "Active"
    assert f["next_due_date"] == "2026-11-23"
    assert f["tenure_remaining"] == "36 months"
    assert "closed_date" not in f
    assert f["outstanding"] == 120000
    assert f["outstanding_display"] == "₹120,000"
    assert f["emi_display"] == "₹8,500"
    assert f["interest_rate"] == 9.5
    assert f["total_interest_paid"] == 95000
    assert f["lender"] == "ICICI Bank"
    assert f["account_number"] == "XXXX5678"
    assert f["late_payment_fee_display"] == "₹750"


def test_loan_context_closed_fields():
    f = build_loan_context(CLOSED, locale="en-US", today=TODAY).facts
    assert f["status"] == "Closed"
    assert f["closed_date"] == "2024-06-15"
    assert "next_due_date" not in f
    assert "tenure_remaining" not in f
    assert f["emi"] == 0


def test_loan_context_has_no_progress_percent():
    f = build_loan_context(ACTIVE, locale="en-US", today=TODAY).facts
    assert not any("progress" in key for key in f)
    assert "id" not in f


def test_loan_shown_history_stats():
    s = build_loan_context(ACTIVE, locale="en-US", today=TODAY).facts["shown_history"]
    assert s["months"] == 6
    assert s["paid"] == 3
    assert s["late"] == 1
    assert s["missed"] == 1
    assert s["pending"] == 1
    assert s["amount_paid_in_shown_history"] == 18000
    assert s["principal_in_shown_history"] == 12000
    assert s["interest_in_shown_history"] == 6000
    assert s["late_months"] == ["September 2026"]
    assert s["missed_months"] == ["August 2026"]
    assert s["pending_months"] == ["June 2026"]
    assert s["amount_paid_in_shown_history_display"] == "₹18,000"


def test_loan_context_includes_vehicle_details_when_present():
    f = build_loan_context(ACTIVE, locale="en-US", today=TODAY).facts
    assert f["vehicle_details"] == {"make": "Honda", "model": "City", "year": 2022, "registration": "MH01AB1234"}
    assert "property_details" not in f


def test_loan_context_includes_property_details_when_present():
    f = build_loan_context(PROPERTY, locale="en-US", today=TODAY).facts
    assert "vehicle_details" not in f
    assert f["property_details"]["address"] == "123 Main Street, Mumbai"
    assert f["property_details"]["property_value_display"] == "₹4,000,000"


def test_loan_history_rows_and_truncation():
    c = build_loan_context(ACTIVE, locale="en-US", today=TODAY, max_history_rows=4)
    assert len(c.rows) == 4
    assert c.rows_total == 6
    assert c.truncated is True
    assert set(c.rows[0]) == {"month_year", "due_date", "paid_date", "amount", "principal", "interest", "status", "amount_display"}
    full = build_loan_context(ACTIVE, locale="en-US", today=TODAY)
    assert full.truncated is False
    assert full.facts["shown_history"]["months"] == 6


def test_section_context_wraps_data():
    c = build_section_context("Dashboard", "credit-score", "Credit Score 750 +12 points", {"score": 750}, locale="en-IN", today=TODAY)
    assert c.page == "section"
    assert c.page_title == "Dashboard › credit-score"
    assert c.facts == {"page": "Dashboard", "section": "credit-score", "section_text": "Credit Score 750 +12 points", "data": {"score": 750}}
    assert c.rows == []
    assert c.rows_total == 0
    assert c.metadata["locale"] == "en-IN"
