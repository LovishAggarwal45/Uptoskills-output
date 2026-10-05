"""Precompiled regular expressions and label dictionaries for structured extraction."""

import re
from typing import Dict, List, Pattern

# =============================================================================
# 1. DATE PATTERNS
# =============================================================================

ISO_DATE_PATTERN: Pattern[str] = re.compile(
    r"\b(?P<year>\d{4})[-/.](?P<month>0[1-9]|1[0-2])[-/.](?P<day>0[1-9]|[12]\d|3[01])\b"
)

DMY_DATE_PATTERN: Pattern[str] = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])[-/.](?P<month>0?[1-9]|1[0-2])[-/.](?P<year>\d{4}|\d{2})\b"
)

MDY_DATE_PATTERN: Pattern[str] = re.compile(
    r"\b(?P<month>0?[1-9]|1[0-2])[-/.](?P<day>0?[1-9]|[12]\d|3[01])[-/.](?P<year>\d{4}|\d{2})\b"
)

TEXT_DATE_DMY_PATTERN: Pattern[str] = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])\s+(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

TEXT_DATE_MDY_PATTERN: Pattern[str] = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[,\s]+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# =============================================================================
# 2. MONEY & CURRENCY PATTERNS
# =============================================================================

CURRENCY_SYMBOLS: Dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "INR": "INR",
    "CAD": "CAD",
    "AUD": "AUD",
    "JPY": "JPY",
    "CHF": "CHF",
}

# Matches strings like $1,250.00, ₹ 1000.50, 1.250,50 EUR, 1.250,50 €, 4500.00
MONEY_PATTERN: Pattern[str] = re.compile(
    r"(?i)(?P<currency>[$€£₹¥]|USD|EUR|GBP|INR|CAD|AUD|JPY)?\s*"
    r"(?P<amount>\d{1,3}(?:\.\d{3})+(?:,\d{1,2})|\d{1,3}(?:[,\s]\d{3})+(?:\.\d{1,2})|\d+\.\d{1,2}|\d+,\d{1,2}|\d{1,3}(?:[,\s.]\d{3})+|\d+)"
    r"\s*(?P<curr_suffix>[$€£₹¥]|USD|EUR|GBP|INR|CAD|AUD|JPY)?",
)

# =============================================================================
# 3. EMAIL & PHONE PATTERNS
# =============================================================================

EMAIL_PATTERN: Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b"
)

# Standard international and national phone formats
PHONE_PATTERN: Pattern[str] = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b"
)

# =============================================================================
# 4. IDENTIFIERS, ORGS & TAX IDs
# =============================================================================

ORG_SUFFIX_PATTERN: Pattern[str] = re.compile(
    r"\b(?:Inc\.?|Corp\.?|Corporation|LLC|Ltd\.?|Limited|GmbH|Pvt\.?\s*Ltd\.?|Group|Enterprises|Solutions|Services|Technologies|Logistics|Co\.?|Company)\b",
    re.IGNORECASE,
)

INVOICE_ID_PATTERN: Pattern[str] = re.compile(
    r"(?i)\b(?:INV|INVOICE)[-:\s#]*([A-Z0-9-]{3,})\b"
)

GENERIC_ID_PATTERN: Pattern[str] = re.compile(
    r"\b[A-Z0-9]{2,4}-[A-Z0-9]{3,8}(?:-[A-Z0-9]{1,6})?\b"
)

TAX_ID_PATTERN: Pattern[str] = re.compile(
    r"(?i)\b(?:VAT|GST|GSTIN|EIN|TIN|SSN)[-:\s#]*([A-Z0-9-]{5,})\b"
)

# =============================================================================
# 5. FIELD LABELS DICTIONARIES
# =============================================================================

INVOICE_FIELD_LABELS: Dict[str, List[str]] = {
    "invoice_number": [
        "invoice number", "invoice no.", "invoice no", "invoice #",
        "inv number", "inv no.", "inv no", "inv #", "bill no", "bill #",
    ],
    "invoice_date": [
        "invoice date", "bill date", "date of issue", "issue date", "billing date", "date:",
    ],
    "due_date": [
        "due date", "payment due", "pay by", "due by", "expiry date",
    ],
    "vendor_name": [
        "vendor:", "vendor", "seller:", "seller", "billed by:", "billed by",
        "company name:", "company:", "supplier:", "supplier", "from:", "issuer:",
        "service provider:",
    ],
    "vendor_address": [
        "vendor address:", "seller address:", "from address:", "supplier address:",
    ],
    "customer_name": [
        "bill to:", "bill to", "billed to:", "billed to", "invoice to:", "invoice to",
        "customer name:", "customer:", "client name:", "client:", "sold to:", "sold to",
        "buyer:", "buyer name:", "recipient:",
    ],
    "customer_address": [
        "billing address:", "ship to:", "shipped to:", "delivery address:", "destination address:",
    ],
    "subtotal": [
        "subtotal", "sub-total", "sub total", "net amount", "total before tax", "taxable amount",
    ],
    "tax": [
        "tax", "vat", "gst", "sales tax", "tax amount", "total tax", "cgst", "sgst", "igst", "utgst",
        "vat amount", "gst amount", "estimated tax",
    ],
    "total": [
        "total due", "amount due", "balance due", "total amount", "grand total", "net payable", "total",
    ],
    "payment_terms": [
        "payment terms", "terms of payment", "due terms",
    ],
    "purchase_order_number": [
        "purchase order", "purchase order #", "purchase order no", "purchase order no.",
        "po number", "po no.", "po no", "po #", "p.o. #", "p.o. no.", "po ref",
    ],
    "tax_id": [
        "tax id", "tax id:", "tax id #", "vat no", "vat no.", "vat number", "vat reg",
        "vat id", "gstin", "gstin:", "gst no", "gst no.", "ein", "tin",
    ],
}

RECEIPT_FIELD_LABELS: Dict[str, List[str]] = {
    "merchant_name": [
        "merchant name:", "merchant:", "store name:", "retailer:",
    ],
    "merchant_address": [
        "store address:", "location:",
    ],
    "receipt_number": [
        "receipt number", "receipt no.", "receipt no", "receipt #", "rcpt #",
        "order number", "order #", "trans #", "transaction #", "check #",
    ],
    "transaction_date": [
        "date:", "trans date:", "order date:", "txn date:", "date",
    ],
    "transaction_time": [
        "time:", "trans time:", "order time:",
    ],
    "cashier": [
        "cashier:", "server:", "operator:", "served by:", "clerk:",
    ],
    "terminal_id": [
        "terminal #", "register #", "pos #", "terminal id", "lane #",
    ],
    "subtotal": [
        "subtotal", "sub total", "sub-total",
    ],
    "tax": [
        "tax", "sales tax", "vat", "gst",
    ],
    "total": [
        "total", "amount", "total paid", "grand total",
    ],
    "payment_method": [
        "cash", "visa", "mastercard", "amex", "debit", "credit card", "payment type:", "tender:",
    ],
}

FORM_FIELD_LABELS: Dict[str, List[str]] = {
    "applicant_name": [
        "applicant name:", "full name:", "name of applicant:", "patient name:", "student name:", "name:",
    ],
    "first_name": [
        "first name:", "given name:", "forename:",
    ],
    "last_name": [
        "last name:", "surname:", "family name:",
    ],
    "date_of_birth": [
        "date of birth:", "dob:", "birth date:", "d.o.b.:",
    ],
    "phone": [
        "phone number:", "phone:", "telephone:", "mobile:", "cell:", "tel:",
    ],
    "email": [
        "email address:", "email:", "e-mail:",
    ],
    "address": [
        "residential address:", "permanent address:", "street address:", "address:",
    ],
    "form_number": [
        "form number:", "form no:", "form #:", "application no:", "app #:", "reference no:",
    ],
    "signature_indicator": [
        "signature of applicant:", "signature:", "sign here:", "applicant signature:",
    ],
}

# =============================================================================
# 6. URL & PROFILES PATTERNS
# =============================================================================

URL_PATTERN: Pattern[str] = re.compile(
    r"\b(?:https?://|www\.)[A-Za-z0-9.\-]+(?::\d+)?(?:/[A-Za-z0-9._%+\-~#=&\?]*)*\b|"
    r"\b(?:linkedin\.com/in/[a-zA-Z0-9_\-]+|github\.com/[a-zA-Z0-9_\-]+)\b",
    re.IGNORECASE,
)

LINKEDIN_PATTERN: Pattern[str] = re.compile(
    r"(?i)\b(?:https?://(?:www\.)?)?linkedin\.com/in/([a-zA-Z0-9_\-]+)\b"
)

GITHUB_PATTERN: Pattern[str] = re.compile(
    r"(?i)\b(?:https?://(?:www\.)?)?github\.com/([a-zA-Z0-9_\-]+)\b"
)

SECTION_HEADER_PATTERN: Pattern[str] = re.compile(
    r"^(?:SUMMARY|EXECUTIVE\s+SUMMARY|OBJECTIVE|CAREER\s+OBJECTIVE|EXPERIENCE|WORK\s+EXPERIENCE|"
    r"PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT\s+HISTORY|EDUCATION|ACADEMIC\s+BACKGROUND|PROJECTS|"
    r"KEY\s+PROJECTS|TECHNICAL\s+SKILLS|SKILLS|CORE\s+COMPETENCIES|CERTIFICATIONS|LICENSES|"
    r"PUBLICATIONS|LANGUAGES|AWARDS|HONORS|REFERENCES|TERMS\s+(?:AND|&)\s+CONDITIONS|NOTES|DECLARATION)[:\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

# =============================================================================
# 7. RESUME & GENERAL FIELD LABELS
# =============================================================================

RESUME_FIELD_LABELS: Dict[str, List[str]] = {
    "candidate_name": [
        "name:", "candidate name:", "applicant name:", "full name:", "curriculum vitae of:",
    ],
    "email": [
        "email:", "email address:", "e-mail:", "mail:",
    ],
    "phone": [
        "phone:", "mobile:", "contact:", "tel:", "telephone:", "cell:",
    ],
    "location": [
        "location:", "address:", "city:", "residence:",
    ],
    "linkedin": [
        "linkedin:", "linkedin profile:", "linkedin.com/in/",
    ],
    "github": [
        "github:", "github profile:", "github.com/",
    ],
    "summary": [
        "summary:", "professional summary:", "profile summary:", "executive summary:", "objective:", "career objective:",
    ],
    "education": [
        "education:", "academic background:", "qualifications:", "academic history:",
    ],
    "experience": [
        "experience:", "work experience:", "professional experience:", "employment history:",
    ],
    "skills": [
        "skills:", "technical skills:", "core competencies:", "technologies:", "tools & technologies:", "programming languages:",
    ],
    "certifications": [
        "certifications:", "certificates:", "courses & certifications:", "licenses:",
    ],
    "projects": [
        "projects:", "key projects:", "academic projects:", "personal projects:",
    ],
}

GENERAL_FIELD_LABELS: Dict[str, List[str]] = {
    "title": [
        "title:", "document title:", "subject:", "re:", "topic:",
    ],
    "author": [
        "author:", "prepared by:", "from:", "issued by:", "submitted by:", "written by:",
    ],
    "recipient": [
        "to:", "prepared for:", "submitted to:", "attention:", "attn:",
    ],
    "date": [
        "date:", "issue date:", "effective date:", "created date:", "published date:",
    ],
    "organization": [
        "organization:", "company:", "institution:", "department:", "agency:",
    ],
    "summary": [
        "summary:", "executive summary:", "abstract:", "overview:",
    ],
}
