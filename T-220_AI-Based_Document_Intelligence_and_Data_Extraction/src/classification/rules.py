"""Structured, explainable classification rules and pattern matching definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

from src.classification.models import ClassificationEvidence, SignalType
from src.core.models import BoundingBox, DocumentPage, OCRTextRegion
from src.core.types import DocumentType


@dataclass
class RuleDefinition:
    """Individual rule definition targeting a specific document class."""
    name: str
    target_type: DocumentType
    signal_type: SignalType
    pattern: str
    weight: float
    is_regex: bool = False
    is_negative: bool = False
    header_only: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.is_regex:
            self._compiled_regex: Optional[Pattern[str]] = re.compile(self.pattern, re.IGNORECASE)
        else:
            # Word-boundary aware matching for keywords and phrases
            escaped = re.escape(self.pattern)
            # Handle cases where pattern starts or ends with alphanumeric
            prefix = r"\b" if re.match(r"^\w", self.pattern) else r""
            suffix = r"\b" if re.search(r"\w$", self.pattern) else r""
            # Normalize internal whitespace in pattern to match any whitespace sequence
            pattern_regex = re.sub(r"\\\s+", r"\\s+", escaped)
            self._compiled_regex = re.compile(f"{prefix}{pattern_regex}{suffix}", re.IGNORECASE)

    def find_matches(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Search text for all matching occurrences.

        Returns:
            List of tuples: (matched_substring, (start_char_idx, end_char_idx))
        """
        if not text or not self._compiled_regex:
            return []

        matches: List[Tuple[str, Tuple[int, int]]] = []
        for match in self._compiled_regex.finditer(text):
            matches.append((match.group(0), (match.start(), match.end())))
        return matches


def get_default_classification_rules() -> List[RuleDefinition]:
    """Compile standard baseline rule collection for all supported document classes."""
    rules: List[RuleDefinition] = []

    # =========================================================================
    # 1. INVOICE RULES
    # =========================================================================
    invoice_signals = [
        # High-confidence phrases
        ("tax invoice", SignalType.PHRASE, 4.0, "Identifies tax invoice header"),
        ("commercial invoice", SignalType.PHRASE, 4.0, "Identifies commercial invoice header"),
        ("proforma invoice", SignalType.PHRASE, 4.0, "Identifies proforma invoice header"),
        ("invoice to", SignalType.PHRASE, 3.5, "Standard billing recipient indicator"),
        ("bill to", SignalType.PHRASE, 3.0, "Billing address prefix"),
        ("billed to", SignalType.PHRASE, 3.0, "Billing address prefix"),
        ("remit to", SignalType.PHRASE, 3.0, "Remittance payment instruction"),
        ("invoice number", SignalType.PHRASE, 3.5, "Explicit invoice identifier label"),
        ("invoice no", SignalType.PHRASE, 3.0, "Abbreviated invoice identifier label"),
        ("invoice #", SignalType.PHRASE, 3.0, "Pound-sign invoice identifier label"),
        ("inv #", SignalType.PHRASE, 2.5, "Short invoice identifier label"),
        ("invoice date", SignalType.PHRASE, 3.0, "Invoice issuance date"),
        ("due date", SignalType.PHRASE, 2.5, "Payment deadline label"),
        ("payment due", SignalType.PHRASE, 2.5, "Payment requirement label"),
        ("amount due", SignalType.PHRASE, 2.5, "Outstanding payment total"),
        ("total due", SignalType.PHRASE, 2.5, "Outstanding payment total"),
        ("balance due", SignalType.PHRASE, 2.5, "Remaining payment balance"),
        ("payment terms", SignalType.PHRASE, 2.5, "Commercial payment conditions"),
        ("purchase order", SignalType.PHRASE, 2.0, "Associated purchase order"),
        ("po number", SignalType.PHRASE, 2.0, "PO reference number"),
        ("po #", SignalType.PHRASE, 2.0, "PO reference number"),
        ("net 30", SignalType.PHRASE, 2.5, "Standard 30-day payment term"),
        ("net 60", SignalType.PHRASE, 2.5, "Standard 60-day payment term"),
        ("net 15", SignalType.PHRASE, 2.0, "Standard 15-day payment term"),
        ("vat number", SignalType.PHRASE, 2.0, "Value Added Tax identification"),
        ("vat no", SignalType.PHRASE, 2.0, "Value Added Tax identification"),
        ("vat reg", SignalType.PHRASE, 2.0, "VAT registration number"),
        ("gst number", SignalType.PHRASE, 2.0, "Goods and Services Tax ID"),
        ("tax rate", SignalType.PHRASE, 1.5, "Tax rate itemization"),
        ("wire transfer", SignalType.PHRASE, 1.5, "B2B wire transfer instructions"),
        ("bank transfer", SignalType.PHRASE, 1.5, "Bank transfer instructions"),
        ("swift", SignalType.KEYWORD, 1.5, "SWIFT international bank code"),
        ("iban", SignalType.KEYWORD, 2.0, "IBAN bank identifier"),
        ("subtotal", SignalType.KEYWORD, 1.5, "Line item aggregate sum"),
        ("unit price", SignalType.PHRASE, 1.5, "Itemized unit cost"),
        ("qty", SignalType.KEYWORD, 1.0, "Item quantity header"),
        ("invoice", SignalType.KEYWORD, 0.5, "Weak invoice indicator; requires supporting evidence"),
    ]
    for text, sig_type, weight, desc in invoice_signals:
        rules.append(
            RuleDefinition(
                name=f"invoice:{text}",
                target_type=DocumentType.INVOICE,
                signal_type=sig_type,
                pattern=text,
                weight=weight,
                description=desc,
            )
        )

    # Invoice Regex patterns
    rules.append(
        RuleDefinition(
            name="invoice:regex_inv_id",
            target_type=DocumentType.INVOICE,
            signal_type=SignalType.REGEX,
            pattern=r"\b(?:INV[-:#\s]+[A-Z0-9][A-Z0-9-]{2,}|INVOICE[-:#\s]+[0-9][A-Z0-9-]{2,})\b",
            weight=3.0,
            is_regex=True,
            description="Matches structured invoice IDs like INV-2026-001",
        )
    )
    rules.append(
        RuleDefinition(
            name="invoice:regex_net_terms",
            target_type=DocumentType.INVOICE,
            signal_type=SignalType.REGEX,
            pattern=r"\bNET\s+(?:10|15|30|45|60|90)\b",
            weight=2.0,
            is_regex=True,
            description="Matches standard commercial NET payment terms",
        )
    )

    # Invoice Negative Signals (penalize if receipt-specific, travel, or form-specific terms appear)
    invoice_negative = [
        ("cashier", 2.5, "Receipt-specific register operator"),
        ("register #", 2.5, "POS register identifier"),
        ("change due", 2.5, "Cash change return indicator"),
        ("cash tendered", 2.5, "Cash tender POS action"),
        ("application form", 3.0, "Form-specific application title"),
        ("please print", 2.0, "Form-specific handwriting prompt"),
        ("electronic reservation slip", 5.0, "Railway reservation slip"),
        ("indian railways", 4.0, "Railway ticket issuer"),
        ("irctc", 4.0, "Railway booking service"),
        ("pnr #", 4.0, "Passenger Name Record"),
        ("pnr no", 4.0, "Passenger Name Record"),
        ("train no", 4.0, "Train service number"),
        ("booked from", 3.5, "Railway booking origin"),
        ("boarding at", 3.5, "Railway boarding point"),
        ("passenger details", 3.5, "Travel ticket passenger section"),
        ("boarding pass", 5.0, "Airline boarding pass"),
        ("flight no", 4.0, "Flight number"),
        ("gate no", 3.5, "Airport boarding gate"),
        ("berth", 3.0, "Train berth allocation"),
    ]
    for text, weight, desc in invoice_negative:
        rules.append(
            RuleDefinition(
                name=f"invoice:negative:{text}",
                target_type=DocumentType.INVOICE,
                signal_type=SignalType.NEGATIVE_SIGNAL,
                pattern=text,
                weight=-weight,
                is_negative=True,
                description=desc,
            )
        )

    # =========================================================================
    # 2. RECEIPT RULES
    # =========================================================================
    receipt_signals = [
        ("sales receipt", SignalType.PHRASE, 4.0, "Sales receipt header"),
        ("store receipt", SignalType.PHRASE, 4.0, "Store receipt header"),
        ("cashier", SignalType.KEYWORD, 3.5, "POS operator"),
        ("register #", SignalType.PHRASE, 3.5, "POS register number"),
        ("terminal #", SignalType.PHRASE, 3.5, "POS terminal number"),
        ("pos terminal", SignalType.PHRASE, 3.5, "Point of sale terminal"),
        ("merchant id", SignalType.PHRASE, 3.0, "Merchant card processing ID"),
        ("merchant #", SignalType.PHRASE, 3.0, "Merchant account number"),
        ("store #", SignalType.PHRASE, 2.5, "Retail store number"),
        ("change due", SignalType.PHRASE, 3.5, "Cash return amount"),
        ("cash tender", SignalType.PHRASE, 3.5, "Cash payment tender"),
        ("cash tendered", SignalType.PHRASE, 3.5, "Cash payment tender"),
        ("card tender", SignalType.PHRASE, 3.0, "Card payment tender"),
        ("visa", SignalType.KEYWORD, 2.0, "Card network tender"),
        ("mastercard", SignalType.KEYWORD, 2.0, "Card network tender"),
        ("amex", SignalType.KEYWORD, 2.0, "Card network tender"),
        ("debit card", SignalType.PHRASE, 2.0, "Debit payment tender"),
        ("credit card", SignalType.PHRASE, 2.0, "Credit payment tender"),
        ("items sold", SignalType.PHRASE, 2.5, "Itemized count"),
        ("subtotal", SignalType.KEYWORD, 1.5, "Receipt subtotal"),
        ("tax", SignalType.KEYWORD, 1.0, "Sales tax"),
        ("total", SignalType.KEYWORD, 1.0, "Transaction total"),
        ("thank you for shopping", SignalType.PHRASE, 3.5, "Retail customer greeting"),
        ("thank you for your business", SignalType.PHRASE, 2.5, "Retail customer greeting"),
        ("please come again", SignalType.PHRASE, 3.5, "Retail customer greeting"),
        ("customer copy", SignalType.PHRASE, 2.5, "Receipt copy designation"),
        ("merchant copy", SignalType.PHRASE, 2.5, "Merchant copy designation"),
        ("guest check", SignalType.PHRASE, 3.5, "Restaurant check"),
        ("table #", SignalType.PHRASE, 3.0, "Restaurant table number"),
        ("dine in", SignalType.PHRASE, 2.5, "Hospitality order type"),
        ("take out", SignalType.PHRASE, 2.5, "Hospitality order type"),
        ("receipt", SignalType.KEYWORD, 2.5, "General receipt indicator"),
    ]
    for text, sig_type, weight, desc in receipt_signals:
        rules.append(
            RuleDefinition(
                name=f"receipt:{text}",
                target_type=DocumentType.RECEIPT,
                signal_type=sig_type,
                pattern=text,
                weight=weight,
                description=desc,
            )
        )

    # Receipt Regex patterns
    rules.append(
        RuleDefinition(
            name="receipt:regex_terminal",
            target_type=DocumentType.RECEIPT,
            signal_type=SignalType.REGEX,
            pattern=r"\b(?:CASHIER|REGISTER|TERMINAL|STORE)\s*#?\s*[:\s]?\s*[A-Z0-9-]+\b",
            weight=3.0,
            is_regex=True,
            description="Matches POS terminal/cashier labels with ID numbers",
        )
    )
    rules.append(
        RuleDefinition(
            name="receipt:regex_masked_card",
            target_type=DocumentType.RECEIPT,
            signal_type=SignalType.REGEX,
            pattern=r"\b(?:[X*]{4}[-\s]?[X*]{4}[-\s]?[X*]{4}[-\s]?\d{4}|\*{4,}\d{4})\b",
            weight=2.5,
            is_regex=True,
            description="Matches masked payment card numbers",
        )
    )

    # Receipt Negative Signals
    receipt_negative = [
        ("bill to", 3.0, "Invoicing B2B billing recipient"),
        ("remit to", 3.0, "Invoicing B2B remittance instructions"),
        ("net 30", 3.0, "Invoicing commercial credit terms"),
        ("purchase order", 2.0, "Invoicing PO reference"),
        ("application form", 3.0, "Form application header"),
    ]
    for text, weight, desc in receipt_negative:
        rules.append(
            RuleDefinition(
                name=f"receipt:negative:{text}",
                target_type=DocumentType.RECEIPT,
                signal_type=SignalType.NEGATIVE_SIGNAL,
                pattern=text,
                weight=-weight,
                is_negative=True,
                description=desc,
            )
        )

    # =========================================================================
    # 3. FORM RULES
    # =========================================================================
    form_signals = [
        ("application form", SignalType.PHRASE, 4.0, "Application form title"),
        ("registration form", SignalType.PHRASE, 4.0, "Registration form title"),
        ("enrollment form", SignalType.PHRASE, 4.0, "Enrollment form title"),
        ("intake form", SignalType.PHRASE, 4.0, "Intake form title"),
        ("claim form", SignalType.PHRASE, 4.0, "Insurance or benefit claim form"),
        ("form no", SignalType.PHRASE, 3.0, "Form reference identifier"),
        ("form #", SignalType.PHRASE, 3.0, "Form reference identifier"),
        ("please print", SignalType.PHRASE, 3.0, "Handwritten entry instruction"),
        ("please fill", SignalType.PHRASE, 3.0, "Form completion instruction"),
        ("fill in", SignalType.PHRASE, 2.5, "Form field prompt"),
        ("applicant name", SignalType.PHRASE, 3.5, "Applicant identity field"),
        ("first name", SignalType.PHRASE, 2.5, "Name component field"),
        ("last name", SignalType.PHRASE, 2.5, "Name component field"),
        ("date of birth", SignalType.PHRASE, 3.0, "DOB personal data field"),
        ("dob:", SignalType.PHRASE, 3.0, "DOB abbreviation label"),
        ("social security", SignalType.PHRASE, 3.0, "SSN identity field"),
        ("ssn:", SignalType.PHRASE, 3.0, "SSN abbreviation label"),
        ("signature:", SignalType.PHRASE, 3.0, "Signature line prompt"),
        ("signature of applicant", SignalType.PHRASE, 4.0, "Explicit applicant signature field"),
        ("marital status", SignalType.PHRASE, 2.5, "Personal demographic field"),
        ("emergency contact", SignalType.PHRASE, 3.0, "Emergency contact section"),
        ("declaration", SignalType.KEYWORD, 2.5, "Legal declaration block"),
        ("section a", SignalType.PHRASE, 2.0, "Form section subdivision"),
        ("section b", SignalType.PHRASE, 2.0, "Form section subdivision"),
        ("official use only", SignalType.PHRASE, 3.5, "Administrative clearance area"),
        ("for office use only", SignalType.PHRASE, 3.5, "Administrative clearance area"),
        ("checkbox", SignalType.KEYWORD, 2.0, "Form input element"),
    ]
    for text, sig_type, weight, desc in form_signals:
        rules.append(
            RuleDefinition(
                name=f"form:{text}",
                target_type=DocumentType.FORM,
                signal_type=sig_type,
                pattern=text,
                weight=weight,
                description=desc,
            )
        )

    # Form Regex patterns
    rules.append(
        RuleDefinition(
            name="form:regex_fill_blanks",
            target_type=DocumentType.FORM,
            signal_type=SignalType.REGEX,
            pattern=r"(?:_{4,}|\[\s*\]|\(\s*\)|:\s*_{2,})",
            weight=2.5,
            is_regex=True,
            description="Matches fill-in-the-blank underscores or checkbox brackets",
        )
    )
    rules.append(
        RuleDefinition(
            name="form:regex_form_section",
            target_type=DocumentType.FORM,
            signal_type=SignalType.REGEX,
            pattern=r"\bSECTION\s+[A-Z0-9]\b",
            weight=2.0,
            is_regex=True,
            description="Matches structured form section headers",
        )
    )

    # Form Negative Signals
    form_negative = [
        ("invoice number", 3.0, "Invoicing identifier"),
        ("tax invoice", 3.5, "Invoice document header"),
        ("bill to", 2.5, "Invoicing billing recipient"),
        ("cashier", 3.0, "Receipt register operator"),
        ("terminal #", 3.0, "POS terminal"),
    ]
    for text, weight, desc in form_negative:
        rules.append(
            RuleDefinition(
                name=f"form:negative:{text}",
                target_type=DocumentType.FORM,
                signal_type=SignalType.NEGATIVE_SIGNAL,
                pattern=text,
                weight=-weight,
                is_negative=True,
                description=desc,
            )
        )

    # =========================================================================
    # 4. RESUME / CV RULES
    # =========================================================================
    resume_signals = [
        ("curriculum vitae", SignalType.PHRASE, 4.5, "Curriculum Vitae header"),
        ("resume", SignalType.KEYWORD, 3.5, "Resume header or title"),
        ("work experience", SignalType.PHRASE, 3.5, "Work history section"),
        ("professional experience", SignalType.PHRASE, 3.5, "Professional experience section"),
        ("employment history", SignalType.PHRASE, 3.5, "Employment history section"),
        ("technical skills", SignalType.PHRASE, 3.0, "Technical skills section"),
        ("core competencies", SignalType.PHRASE, 3.0, "Skills and competencies"),
        ("education", SignalType.KEYWORD, 2.5, "Academic education section"),
        ("academic background", SignalType.PHRASE, 3.0, "Academic history"),
        ("projects", SignalType.KEYWORD, 2.0, "Project portfolio section"),
        ("certifications", SignalType.KEYWORD, 2.5, "Professional credentials"),
        ("summary of qualifications", SignalType.PHRASE, 3.0, "Professional summary"),
        ("career objective", SignalType.PHRASE, 3.0, "Career objective section"),
        ("bachelor of", SignalType.PHRASE, 3.0, "Undergraduate degree"),
        ("master of", SignalType.PHRASE, 3.0, "Graduate degree"),
        ("ph.d.", SignalType.KEYWORD, 3.0, "Doctoral degree"),
        ("doctor of philosophy", SignalType.PHRASE, 3.0, "Doctoral degree"),
        ("university", SignalType.KEYWORD, 1.5, "Educational institution"),
        ("gpa", SignalType.KEYWORD, 2.0, "Grade point average"),
        ("cgpa", SignalType.KEYWORD, 2.0, "Cumulative grade point average"),
        ("publications", SignalType.KEYWORD, 2.0, "Research publications"),
        ("programming languages", SignalType.PHRASE, 2.5, "Software skills"),
        ("github.com", SignalType.PHRASE, 2.5, "GitHub profile link"),
        ("linkedin.com/in", SignalType.PHRASE, 3.0, "LinkedIn profile link"),
    ]
    for text, sig_type, weight, desc in resume_signals:
        rules.append(
            RuleDefinition(
                name=f"resume:{text}",
                target_type=DocumentType.RESUME,
                signal_type=sig_type,
                pattern=text,
                weight=weight,
                description=desc,
            )
        )

    # Resume Regex patterns
    rules.append(
        RuleDefinition(
            name="resume:regex_section_headers",
            target_type=DocumentType.RESUME,
            signal_type=SignalType.REGEX,
            pattern=r"\b(?:WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS)\b",
            weight=3.0,
            is_regex=True,
            description="Matches canonical resume section headings",
        )
    )
    rules.append(
        RuleDefinition(
            name="resume:regex_profiles",
            target_type=DocumentType.RESUME,
            signal_type=SignalType.REGEX,
            pattern=r"\b(?:linkedin\.com/in/[a-zA-Z0-9_\-]+|github\.com/[a-zA-Z0-9_\-]+)\b",
            weight=3.5,
            is_regex=True,
            description="Matches developer and professional social profiles",
        )
    )

    # Resume Negative Signals
    resume_negative = [
        ("tax invoice", 4.0, "Invoice document header"),
        ("invoice number", 3.5, "Invoice identifier"),
        ("amount due", 3.0, "Invoice payment total"),
        ("cashier", 3.5, "Receipt operator"),
        ("change due", 3.5, "Receipt cash register"),
        ("remit to", 3.0, "Invoicing B2B remittance"),
        ("official use only", 2.5, "Administrative intake form"),
    ]
    for text, weight, desc in resume_negative:
        rules.append(
            RuleDefinition(
                name=f"resume:negative:{text}",
                target_type=DocumentType.RESUME,
                signal_type=SignalType.NEGATIVE_SIGNAL,
                pattern=text,
                weight=-weight,
                is_negative=True,
                description=desc,
            )
        )

    # =========================================================================
    # 5. GENERAL DOCUMENT RULES
    # =========================================================================
    general_signals = [      
        ("electronic reservation slip", SignalType.PHRASE, 5.0, "Railway reservation ticket header"),
        ("indian railways", SignalType.PHRASE, 3.0, "Railway ticket issuer"),
        ("irctc", SignalType.KEYWORD, 2.0, "Railway ticket booking service"),
        ("pnr", SignalType.KEYWORD, 1.5, "Passenger reservation reference"),
        ("executive summary", SignalType.PHRASE, 3.5, "Formal report summary"),
        ("table of contents", SignalType.PHRASE, 3.5, "Document table of contents"),
        ("introduction", SignalType.KEYWORD, 2.5, "Document introductory section"),
        ("conclusion", SignalType.KEYWORD, 2.5, "Document concluding section"),
        ("abstract", SignalType.KEYWORD, 3.0, "Academic or technical abstract"),
        ("background", SignalType.KEYWORD, 2.0, "Background information section"),
        ("memorandum", SignalType.KEYWORD, 3.5, "Internal business memo header"),
        ("memo", SignalType.KEYWORD, 3.0, "Short business memo header"),
        ("subject:", SignalType.PHRASE, 2.0, "Formal letter/memo subject"),
        ("dear ", SignalType.PHRASE, 2.5, "Letter salutation"),
        ("sincerely", SignalType.KEYWORD, 2.5, "Letter valediction"),
        ("best regards", SignalType.PHRASE, 2.5, "Correspondence sign-off"),
        ("kind regards", SignalType.PHRASE, 2.5, "Correspondence sign-off"),
        ("annual report", SignalType.PHRASE, 3.5, "Annual company report"),
        ("white paper", SignalType.PHRASE, 3.5, "White paper publication"),
        ("meeting minutes", SignalType.PHRASE, 3.5, "Meeting record"),
        ("agenda", SignalType.KEYWORD, 2.5, "Meeting agenda"),
        ("overview", SignalType.KEYWORD, 2.0, "Section overview"),
        ("appendix", SignalType.KEYWORD, 2.5, "Document appendix"),
        ("terms and conditions", SignalType.PHRASE, 3.0, "Legal terms section"),
        ("certificate of", SignalType.PHRASE, 3.5, "Formal certificate title"),
        ("confidentiality agreement", SignalType.PHRASE, 3.5, "Legal agreement title"),
        ("statement of work", SignalType.PHRASE, 3.5, "Formal project SOW"),
        ("user guide", SignalType.PHRASE, 3.0, "Technical user guide"),
    ]
    for text, sig_type, weight, desc in general_signals:
        rules.append(
            RuleDefinition(
                name=f"general_document:{text}",
                target_type=DocumentType.GENERAL_DOCUMENT,
                signal_type=sig_type,
                pattern=text,
                weight=weight,
                description=desc,
            )
        )

    # General Document Regex patterns
    rules.append(
        RuleDefinition(
            name="general_document:regex_memo_header",
            target_type=DocumentType.GENERAL_DOCUMENT,
            signal_type=SignalType.REGEX,
            pattern=r"\b(?:MEMORANDUM|EXECUTIVE\s+SUMMARY|TABLE\s+OF\s+CONTENTS|STATEMENT\s+OF\s+WORK)\b",
            weight=3.0,
            is_regex=True,
            description="Matches formal memo/report header blocks",
        )
    )

    # General Document Negative Signals
    general_negative = [
        ("tax invoice", 3.0, "Invoice document header"),
        ("invoice number", 2.0, "Invoice identifier"),
        ("cashier", 2.5, "Receipt operator"),
        ("application form", 2.5, "Form application header"),
        ("curriculum vitae", 3.0, "Resume document header"),
    ]
    for text, weight, desc in general_negative:
        rules.append(
            RuleDefinition(
                name=f"general_document:negative:{text}",
                target_type=DocumentType.GENERAL_DOCUMENT,
                signal_type=SignalType.NEGATIVE_SIGNAL,
                pattern=text,
                weight=-weight,
                is_negative=True,
                description=desc,
            )
        )

    return rules


def locate_text_provenance(
    matched_text: str,
    page: DocumentPage,
) -> Tuple[Optional[BoundingBox], Optional[float]]:
    """Attempt to locate spatial bounding box and OCR confidence for matched text in a page.

    Args:
        matched_text: Substring that matched the classification rule.
        page: DocumentPage containing OCR text regions and words.

    Returns:
        Tuple of (merged_bounding_box, mean_ocr_confidence).
    """
    if not page.ocr_text_regions or not matched_text:
        return None, None

    matched_lower = matched_text.lower().strip()
    matched_words = matched_lower.split()
    if not matched_words:
        return None, None

    # Search at the word level across ocr_text_regions
    candidate_boxes: List[BoundingBox] = []
    candidate_confs: List[float] = []

    for region in page.ocr_text_regions:
        reg_text_lower = region.text.lower().strip()
        # Direct exact or substring match in region
        if matched_lower in reg_text_lower:
            if region.bounding_box:
                candidate_boxes.append(region.bounding_box)
            if region.confidence is not None:
                candidate_confs.append(region.confidence)

    # If nothing matched in top regions, check sub_regions recursively
    if not candidate_boxes:
        for region in page.ocr_text_regions:
            for sub in region.sub_regions:
                sub_lower = sub.text.lower().strip()
                if any(w in sub_lower for w in matched_words):
                    if sub.bounding_box:
                        candidate_boxes.append(sub.bounding_box)
                    if sub.confidence is not None:
                        candidate_confs.append(sub.confidence)

    if not candidate_boxes:
        return None, None

    # Merge bounding boxes
    xmins = [b.xmin for b in candidate_boxes]
    ymins = [b.ymin for b in candidate_boxes]
    xmaxs = [b.xmax for b in candidate_boxes]
    ymaxs = [b.ymax for b in candidate_boxes]
    merged_box = BoundingBox(
        xmin=min(xmins),
        ymin=min(ymins),
        xmax=max(xmaxs),
        ymax=max(ymaxs),
        is_normalized=candidate_boxes[0].is_normalized,
    )
    mean_conf = sum(candidate_confs) / len(candidate_confs) if candidate_confs else None

    return merged_box, mean_conf
