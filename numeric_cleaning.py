"""
DataLens — Messy Numeric Cleaning
--------------------------------------
Real-world numeric columns often aren't stored as clean numbers:
  - Currency: "$1,234.56", "€999", "£45.00"
  - Thousand-separators: "1,234,567"
  - Percentages: "45%"
  - Accounting-style negatives: "(123.45)" meaning -123.45

pandas reads all of these as plain text, since they contain non-digit
characters. Left alone, a column like this silently becomes "text" or
"categorical" instead of numeric — no crash, just wrong, quiet
analysis (no mean/median, no charts, broken correlations).

This module detects and cleans these patterns BEFORE column type
classification, so a "$1,234.56" column is correctly treated as
numeric with the value 1234.56, not as an uninterpretable string.

Run: python numeric_cleaning.py (self-tests)
Requires: pip install pandas
"""

import re

import pandas as pd

CURRENCY_SYMBOLS = r"[$€£¥₹]"
MESSY_NUMERIC_PATTERN = re.compile(
    rf"^\s*\(?\s*{CURRENCY_SYMBOLS}?\s*-?[\d,]+\.?\d*\s*%?\s*\)?\s*$"
)


def clean_numeric_string(value) -> float:
    """Parse a single messy numeric string into a float. Returns NaN
    for values that don't actually match the messy-numeric pattern,
    rather than raising — callers check the overall column match rate
    before trusting this, so an occasional real non-numeric value
    (e.g. a stray text note in an otherwise numeric column) just
    becomes a missing value rather than crashing the whole load."""
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    is_negative_accounting = text.startswith("(") and text.endswith(")")
    if is_negative_accounting:
        text = text[1:-1].strip()

    is_percentage = text.endswith("%")
    if is_percentage:
        text = text[:-1].strip()

    text = re.sub(CURRENCY_SYMBOLS, "", text)
    text = text.replace(",", "")
    text = text.strip()

    try:
        result = float(text)
    except ValueError:
        return float("nan")

    if is_negative_accounting:
        result = -abs(result)

    # Deliberate design choice: percentages keep their written magnitude
    # (e.g. "45%" -> 45.0, not 0.45). Silently rescaling would change
    # the numbers a user sees relative to their source file — safer to
    # preserve what was written and let the column name/context carry
    # the "this is a percentage" meaning.
    return result


def is_messy_numeric_column(series: pd.Series, min_match_rate: float = 0.8) -> bool:
    """Check whether a text column is actually a numeric column in
    disguise. Requires most non-null values to match the messy-numeric
    pattern — not literally every value, since a column that's 95%
    clean currency values with one stray "N/A" is still meaningfully
    numeric."""
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False

    matches = non_null.apply(lambda v: bool(MESSY_NUMERIC_PATTERN.match(v)))
    return matches.mean() >= min_match_rate


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert a detected messy-numeric column into an actual numeric
    pandas Series."""
    return series.apply(clean_numeric_string)


if __name__ == "__main__":
    print("=== Self-test: individual value cleaning ===")
    test_cases = [
        ("$1,234.56", 1234.56),
        ("€999", 999.0),
        ("£45.00", 45.0),
        ("1,234,567", 1234567.0),
        ("45%", 45.0),
        ("(123.45)", -123.45),
        ("-50", -50.0),
        ("100", 100.0),
        ("", None),  # NaN, checked separately
    ]
    for input_val, expected in test_cases:
        result = clean_numeric_string(input_val)
        if expected is None:
            assert pd.isna(result), f"Expected NaN for {input_val!r}, got {result}"
            print(f"  {input_val!r:15} -> NaN (correct)")
        else:
            assert abs(result - expected) < 0.001, f"Expected {expected} for {input_val!r}, got {result}"
            print(f"  {input_val!r:15} -> {result} (correct)")
    print("PASSED\n")

    print("=== Self-test: non-numeric text should NOT be treated as numeric ===")
    non_numeric_cases = ["Alice", "New York", "Category A", "2026-01-15"]
    for val in non_numeric_cases:
        result = clean_numeric_string(val)
        assert pd.isna(result), f"Expected NaN for non-numeric {val!r}, got {result}"
        print(f"  {val!r:15} -> NaN (correctly rejected)")
    print("PASSED\n")

    print("=== Self-test: column-level detection ===")
    messy_currency_col = pd.Series(["$1,234.56", "$999.00", "$45.50", "$2,100.00"])
    assert is_messy_numeric_column(messy_currency_col), "Should detect currency column as numeric"
    print("  Currency column correctly detected as numeric")

    normal_categorical_col = pd.Series(["Alice", "Bob", "Charlie", "Dave"])
    assert not is_messy_numeric_column(normal_categorical_col), "Should NOT detect names as numeric"
    print("  Name column correctly NOT detected as numeric")

    mixed_col = pd.Series(["$100", "$200", "N/A", "$300", "$400"])
    assert is_messy_numeric_column(mixed_col), "Should still detect as numeric with one stray non-numeric value (80% threshold)"
    print("  Mostly-numeric column with one stray value still correctly detected")
    print("PASSED\n")

    print("=== Self-test: full column cleaning ===")
    df = pd.DataFrame({"revenue": ["$1,234.56", "$999.00", "N/A", "$2,100.00"]})
    cleaned = clean_numeric_column(df["revenue"])
    print(cleaned.tolist())
    assert cleaned.iloc[0] == 1234.56
    assert pd.isna(cleaned.iloc[2])
    print("PASSED\n")

    print("All self-tests passed.")
