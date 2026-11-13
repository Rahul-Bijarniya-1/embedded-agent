"""Money formatting that matches what the page renders with toLocaleString()."""
from __future__ import annotations

RUPEE = "₹"


def is_indian_locale(locale: str) -> bool:
    loc = (locale or "").strip().lower().replace("_", "-")
    return loc == "hi" or loc.startswith("hi-") or loc.endswith("-in")


def _group_indian(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def format_money(amount: float, locale: str) -> str:
    value = float(amount)
    negative = value < 0
    value = abs(value)
    if abs(value - round(value)) < 1e-9:
        whole, fraction = str(int(round(value))), ""
    else:
        whole, decimals = f"{value:.2f}".split(".")
        fraction = "." + decimals
    grouped = _group_indian(whole) if is_indian_locale(locale) else f"{int(whole):,}"
    return f"{'-' if negative else ''}{RUPEE}{grouped}{fraction}"
