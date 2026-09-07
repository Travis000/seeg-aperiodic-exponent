"""How a p value is rendered in the manuscript.

This is included because rounding is a place where a number can quietly change
meaning, and doing it in code rather than by hand is the only way to be sure it
was done the same way in all 23 places the paper reports one.

The rule:

======================  ===============================
``p >= 0.01``           two decimals
``0.001 <= p < 0.01``   three decimals
``p < 0.001``           rendered as ``p < 0.001``
======================  ===============================

with one deliberate exception, described in :func:`format_p`.
"""

from __future__ import annotations

import decimal

#: Values in this window keep three decimals regardless of the table above.
BOUNDARY_WINDOW = (decimal.Decimal("0.045"), decimal.Decimal("0.055"))


def round_p(p: float) -> str | None:
    """Round a p value for display. ``None`` means "render as ``p < 0.001``".

    Takes the value from the analysis, not a previously rounded string. That
    distinction is the whole reason this is a function: rounding 0.0345 to
    three decimals and then rounding the *string* to two gives 0.04, whereas
    the correct answer is 0.03. Rounding twice is not the same as rounding
    once, and doing it by hand across a manuscript is how it happens.

    The exception: a value between 0.045 and 0.055 keeps three decimals. Under
    the plain rule 0.046 and 0.054 both print as "0.05", so a result on either
    side of the conventional threshold would be displayed identically. Keeping
    the third digit lets the reader see which side it fell on, rather than
    having the rounding make that decision invisibly.
    """
    value = decimal.Decimal(repr(float(p)))
    if value < decimal.Decimal("0.001"):
        return None

    in_boundary_window = BOUNDARY_WINDOW[0] <= value < BOUNDARY_WINDOW[1]
    places = (decimal.Decimal("0.001")
              if value < decimal.Decimal("0.01") or in_boundary_window
              else decimal.Decimal("0.01"))
    return str(value.quantize(places, rounding=decimal.ROUND_HALF_UP))


def format_p(p: float, lead: str = "p = ") -> str:
    """``round_p`` with the leading text, e.g. ``"p = 0.03"``, ``"p < 0.001"``."""
    rounded = round_p(p)
    return "p < 0.001" if rounded is None else f"{lead}{rounded}"


def format_ci(low: float, high: float, digits: int = 2) -> str:
    """``"[-0.91, -0.44]"`` -- intervals accompany every effect estimate."""
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"
