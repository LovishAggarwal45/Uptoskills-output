"""Data normalization functions for extracted text, currency, dates, and identifiers."""

import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.extraction.patterns import (
    CURRENCY_SYMBOLS,
    DMY_DATE_PATTERN,
    EMAIL_PATTERN,
    ISO_DATE_PATTERN,
    MDY_DATE_PATTERN,
    MONEY_PATTERN,
    MONTH_MAP,
    PHONE_PATTERN,
    TEXT_DATE_DMY_PATTERN,
    TEXT_DATE_MDY_PATTERN,
)


def normalize_text(text: Optional[str]) -> str:
    """Normalize raw text by collapsing whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_money(
    raw_value: Optional[str],
) -> Tuple[Optional[float], Optional[str]]:
    """Parse monetary text into a numeric amount and optional currency."""
    if not raw_value:
        return None, None

    cleaned = normalize_text(raw_value)

    if re.fullmatch(r"\(?\s*@?\s*\d+(?:\.\d+)?%\s*\)?", cleaned):
        return None, None

    cleaned_no_rate = re.sub(
        r"\(?\s*@?\s*\d+(?:\.\d+)?%\s*\)?", "", cleaned
    ).strip()

    cleaned_no_rate = re.sub(
        r"^(?:tax|sales\s*tax|vat|gst|cgst|sgst|igst|utgst|"
        r"subtotal|sub-total|total|amount|balance\s*due|net)\b",
        "",
        cleaned_no_rate,
        flags=re.IGNORECASE,
    ).strip()

    cleaned_no_rate = re.sub(r"^[:\s\-]+", "", cleaned_no_rate).strip()
    cleaned_no_rate = re.sub(r"[:\s\-]+$", "", cleaned_no_rate).strip()
    target_text = cleaned_no_rate if cleaned_no_rate else cleaned

    non_curr_text = re.sub(
        r"(?i)\b(?:USD|EUR|GBP|INR|CAD|AUD|JPY|CHF)\b",
        "",
        target_text,
    )
    non_curr_text = re.sub(r"[$€£₹¥\d\s.,\-+()]", "", non_curr_text)

    if non_curr_text.strip():
        return None, None

    match = MONEY_PATTERN.search(target_text)
    if not match:
        return None, None

    curr_prefix = match.group("currency")
    curr_suffix = match.group("curr_suffix")
    amount_str = match.group("amount")

    if not amount_str:
        return None, None

    currency: Optional[str] = None
    if curr_prefix:
        currency = CURRENCY_SYMBOLS.get(curr_prefix.upper(), curr_prefix)
    elif curr_suffix:
        currency = CURRENCY_SYMBOLS.get(curr_suffix.upper(), curr_suffix)

    amount_str = amount_str.replace(" ", "")

    try:
        if "," in amount_str and "." in amount_str:
            if amount_str.rfind(",") > amount_str.rfind("."):
                amount_str = amount_str.replace(".", "").replace(",", ".")
            else:
                amount_str = amount_str.replace(",", "")
        elif "," in amount_str:
            parts = amount_str.split(",")
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                amount_str = amount_str.replace(",", ".")
            else:
                amount_str = amount_str.replace(",", "")

        return float(amount_str), currency
    except (ValueError, TypeError):
        return None, currency


def normalize_date(
    raw_value: Optional[str],
    default_order: str = "DMY",
) -> Tuple[Optional[str], bool]:
    """Normalize a date to YYYY-MM-DD and report numeric ambiguity."""
    if not raw_value:
        return None, False

    cleaned = normalize_text(raw_value)

    iso_match = ISO_DATE_PATTERN.search(cleaned)
    if iso_match:
        y = int(iso_match.group("year"))
        m = int(iso_match.group("month"))
        d = int(iso_match.group("day"))
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d"), False
        except ValueError:
            return None, False

    text_dmy = TEXT_DATE_DMY_PATTERN.search(cleaned)
    if text_dmy:
        d = int(text_dmy.group("day"))
        m = MONTH_MAP.get(text_dmy.group("month").lower())
        y = int(text_dmy.group("year"))
        if m:
            try:
                return datetime(y, m, d).strftime("%Y-%m-%d"), False
            except ValueError:
                return None, False

    text_mdy = TEXT_DATE_MDY_PATTERN.search(cleaned)
    if text_mdy:
        m = MONTH_MAP.get(text_mdy.group("month").lower())
        d = int(text_mdy.group("day"))
        y = int(text_mdy.group("year"))
        if m:
            try:
                return datetime(y, m, d).strftime("%Y-%m-%d"), False
            except ValueError:
                return None, False

    dmy_match = DMY_DATE_PATTERN.search(cleaned)
    if dmy_match:
        v1 = int(dmy_match.group("day"))
        v2 = int(dmy_match.group("month"))
        y_str = dmy_match.group("year")
        y_num = int(y_str)
        y = y_num if len(y_str) == 4 else (
            2000 + y_num if y_num < 50 else 1900 + y_num
        )

        if v1 > 12 and v2 <= 12:
            d, m = v1, v2
            ambiguous = False
        elif v2 > 12 and v1 <= 12:
            d, m = v2, v1
            ambiguous = False
        elif v1 <= 12 and v2 <= 12:
            if default_order.upper() == "MDY":
                m, d = v1, v2
            else:
                d, m = v1, v2
            ambiguous = True
        else:
            return None, False

        try:
            return datetime(y, m, d).strftime("%Y-%m-%d"), ambiguous
        except ValueError:
            return None, False

    return None, False


def normalize_email(raw_value: Optional[str]) -> Optional[str]:
    """Validate and normalize an email address."""
    if not raw_value:
        return None

    match = EMAIL_PATTERN.search(raw_value)
    if match:
        return match.group(0).lower().strip()
    return None


def normalize_phone(raw_value: Optional[str]) -> Optional[str]:
    """Normalize a phone number while preserving a leading plus sign."""
    if not raw_value:
        return None

    match = PHONE_PATTERN.search(raw_value)
    if match:
        cleaned = normalize_text(match.group(0))
        has_plus = cleaned.startswith("+")
        digits = re.sub(r"\D", "", cleaned)

        if len(digits) >= 7:
            return f"+{digits}" if has_plus else digits

    return None


def normalize_identifier(raw_value: Optional[str]) -> str:
    """Remove common label prefixes and surrounding punctuation from an ID."""
    if not raw_value:
        return ""

    cleaned = normalize_text(raw_value)

    cleaned = re.sub(
        r"^(?:purchase\s*order|p\.?o\.?\s*#|p\.?o\.?\s+no\.?|"
        r"p\.?o\.?\s+number|p\.?o\.?\s*:|po\s*#|po\s+no\.?|"
        r"po\s+num(?:ber)?|po\s*:|invoice\s*#|invoice\s+no\.?|"
        r"invoice\s+number|invoice\s*:|inv\s*#|inv\s+no\.?|"
        r"inv\s*:|tax\s*id\s*:?|vat\s*no\.?|vat\s*reg|vat\s*id|"
        r"gstin\s*no\.?|gstin\s*:?|tin\s*:?|ein\s*:?)"
        r"\s*[:#\-.\s]+\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"^[:#\s\-\./\\]+", "", cleaned)
    cleaned = re.sub(r"[:#\s\-\./\\]+$", "", cleaned)
    return cleaned


def isolate_tax_identifier(text: Optional[str]) -> Optional[str]:
    """Extract a recognizable tax identifier; never guess from arbitrary text.

    Supported formats:
    - Indian GSTIN: 15-character GSTIN format.
    - US EIN: NN-NNNNNNN.
    - Explicitly labelled VAT, GST, GSTIN, or TIN identifiers.
    - EU/UK-style VAT identifiers: country code plus 8-12 alphanumeric chars.

    Returns None if no supported tax identifier is found.
    """
    if not text:
        return None

    cleaned = normalize_text(text).upper()

    # 1. Indian GSTIN.
    gstin_match = re.search(
        r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b",
        cleaned,
    )
    if gstin_match:
        return gstin_match.group(1)

    
    # 2. US EIN, including the project's explicit US-EIN format.
    prefixed_ein_match = re.search(
        r"\bUS-EIN-(\d{8,9})\b",
        cleaned,
        re.IGNORECASE,
    )
    if prefixed_ein_match:
        return f"US-EIN-{prefixed_ein_match.group(1)}"

    ein_match = re.search(
        r"\b(?:EIN[:\s#]*)?(\d{2}-\d{7})\b",
        cleaned,
        re.IGNORECASE,
    )
    if ein_match:
        return ein_match.group(1)


    # 3. Explicitly labelled tax identifiers.
    # Require a label followed by a separator and a compact ID token.
    labelled_match = re.search(
        r"\b(?:VAT|GSTIN|GST|TIN)\s*[:#-]\s*"
        r"([A-Z0-9][A-Z0-9-]{4,19})\b",
        cleaned,
        re.IGNORECASE,
    )
    if labelled_match:
        candidate = labelled_match.group(1).upper()
        excluded = {
            "INVOICE",
            "INVOICENO",
            "TOTAL",
            "AMOUNT",
            "CURRENCY",
            "PAYMENT",
            "SUBTOTAL",
        }

        if candidate not in excluded and re.search(r"\d", candidate):
            return candidate

    
    # 4. EU/UK-style VAT identifiers, including VAT Registration labels.
    vat_match = re.search(
        r"\b(?:VAT\s+REGISTRATION|VAT\s+NUMBER|VAT\s+NO\.?|VAT)"
        r"\s*[:#-]?\s*([A-Z]{2}[0-9A-Z]{8,12})\b",
        cleaned,
        re.IGNORECASE,
    )
    if vat_match:
        candidate = vat_match.group(1).upper()
        if re.search(r"\d", candidate):
            return candidate

    # Also support EU/UK VAT identifiers appearing without a label.
    vat_match = re.search(
        r"\b([A-Z]{2}[0-9A-Z]{8,12})\b",
        cleaned,
        re.IGNORECASE,
    )
    if vat_match:
        candidate = vat_match.group(1).upper()
        if re.search(r"\d", candidate) and candidate not in {
            "INVOICENO", "CUSTOMER", "PURCHASE"
        }:
            return candidate

    # IMPORTANT:
    # Do not return arbitrary text just because it contains digits.
    # Vendor names, addresses, invoice numbers, dates, and amounts
    # are not sufficient evidence of a tax identifier.
    return None