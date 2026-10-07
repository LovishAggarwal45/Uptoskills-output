"""Structured field extractor for Receipt and POS transaction documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import ExtractionConfig
from src.core.models import Document, ExtractedField, Provenance
from src.core.types import (
    ConfidenceSource,
    ExtractionMethod,
    FieldType,
    ValidationStatus,
)
from src.extraction.confidence import calculate_extraction_confidence
from src.extraction.models import ExtractionCandidate
from src.extraction.normalizers import (
    normalize_date,
    normalize_identifier,
    normalize_money,
    normalize_text,
)
from src.extraction.patterns import (
    RECEIPT_FIELD_LABELS,
)
from src.extraction.spatial import (
    find_label_value_matches,
    locate_value_in_regions,
)

TIME_PATTERN = re.compile(
    r"\b(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b"
)


class ReceiptExtractor:
    """Extracts canonical point-of-sale receipt fields with provenance preservation."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    def extract(self, document: Document) -> Dict[str, ExtractedField]:
        """Extract receipt-specific fields from all document pages."""
        extracted_fields: Dict[str, ExtractedField] = {}

        # 1. Merchant Name & Address
        merchant_name, merchant_addr = self._extract_merchant_info(document)
        if merchant_name:
            extracted_fields["merchant_name"] = merchant_name
        if merchant_addr:
            extracted_fields["merchant_address"] = merchant_addr

        # 2. Receipt Number
        rcpt_num = self._extract_receipt_number(document)
        if rcpt_num:
            extracted_fields["receipt_number"] = rcpt_num

        # 3. Transaction Date
        trans_date = self._extract_transaction_date(document)
        if trans_date:
            extracted_fields["transaction_date"] = trans_date

        # 4. Transaction Time
        trans_time = self._extract_transaction_time(document)
        if trans_time:
            extracted_fields["transaction_time"] = trans_time

        # 5. Cashier & Terminal / Register
        cashier_f = self._extract_text_field(document, "cashier", RECEIPT_FIELD_LABELS["cashier"])
        if cashier_f:
            extracted_fields["cashier"] = cashier_f

        terminal_f = self._extract_text_field(document, "terminal_id", RECEIPT_FIELD_LABELS["terminal_id"], FieldType.IDENTIFIER)
        if terminal_f:
            extracted_fields["terminal_id"] = terminal_f

        # 6. Monetary Amounts: Subtotal, Tax, Total
        subtotal_f = self._extract_money_field(document, "subtotal", RECEIPT_FIELD_LABELS["subtotal"])
        if subtotal_f:
            extracted_fields["subtotal"] = subtotal_f

        tax_f = self._extract_money_field(document, "tax", RECEIPT_FIELD_LABELS["tax"])
        if tax_f:
            extracted_fields["tax"] = tax_f

        total_f = self._extract_money_field(document, "total", RECEIPT_FIELD_LABELS["total"])
        if total_f:
            extracted_fields["total"] = total_f

        # 7. Payment Method
        payment_f = self._extract_payment_method(document)
        if payment_f:
            extracted_fields["payment_method"] = payment_f

        # 8. Currency
        curr_f = self._infer_currency(extracted_fields, document)
        if curr_f:
            extracted_fields["currency"] = curr_f

        return extracted_fields

    def _extract_merchant_info(
        self,
        document: Document,
    ) -> Tuple[Optional[ExtractedField], Optional[ExtractedField]]:
        if not document.pages:
            return None, None
        page = document.pages[0]

        # Check label-value first
        matches = find_label_value_matches(page, RECEIPT_FIELD_LABELS["merchant_name"], self.config.spatial)
        if matches:
            m = matches[0]
            val = normalize_text(m.value_text)
            prov = Provenance(
                document_id=document.id,
                page_number=page.page_number,
                raw_text=m.source_line,
                bounding_box=m.value_bbox,
                ocr_confidence=m.mean_ocr_confidence,
            )
            field_obj = ExtractedField(
                name="merchant_name",
                value=val,
                normalized_value=val,
                field_type=FieldType.ORGANIZATION,
                provenance=prov,
                extraction_confidence=0.90,
                is_required=True,
                matched_label=m.label_text,
            )
            return field_obj, None

        # Prominent top line heuristic
        lines = [normalize_text(l) for l in page.raw_text.split("\n") if normalize_text(l)]
        for line in lines[:2]:
            if "receipt" in line.lower() and len(line) < 15:
                continue
            if len(line) >= 3 and not re.search(r"^(date|time|tel|store #|cashier)", line.lower()):
                box, ocr_c = locate_value_in_regions(line, page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=line,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                    extraction_method=ExtractionMethod.LAYOUT_ANALYSIS,
                )
                field_obj = ExtractedField(
                    name="merchant_name",
                    value=line,
                    normalized_value=line,
                    field_type=FieldType.ORGANIZATION,
                    provenance=prov,
                    extraction_confidence=0.78,
                    is_required=True,
                )
                return field_obj, None

        return None, None

    def _extract_receipt_number(self, document: Document) -> Optional[ExtractedField]:
        candidates: List[ExtractionCandidate] = []
        for page in document.pages:
            matches = find_label_value_matches(page, RECEIPT_FIELD_LABELS["receipt_number"], self.config.spatial)
            for m in matches:
                norm_val = normalize_identifier(m.value_text)
                if norm_val:
                    conf = calculate_extraction_confidence(
                        label_match_strength=1.0,
                        pattern_validity_score=0.90,
                        spatial_score=m.spatial_score,
                        candidate_count=len(matches),
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    candidates.append(
                        ExtractionCandidate(
                            value=norm_val,
                            raw_value=m.value_text,
                            normalized_value=norm_val,
                            confidence=conf,
                            source_text=m.source_line,
                            page_number=page.page_number,
                            bounding_box=m.value_bbox,
                            ocr_confidence=m.mean_ocr_confidence,
                            matched_label=m.label_text,
                        )
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        winner = candidates[0]
        prov = Provenance(
            document_id=document.id,
            page_number=winner.page_number,
            raw_text=winner.source_text,
            bounding_box=winner.bounding_box,
            ocr_confidence=winner.ocr_confidence,
        )
        return ExtractedField(
            name="receipt_number",
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.IDENTIFIER,
            provenance=prov,
            extraction_confidence=winner.confidence,
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
        )

    def _extract_transaction_date(self, document: Document) -> Optional[ExtractedField]:
        candidates: List[ExtractionCandidate] = []
        for page in document.pages:
            matches = find_label_value_matches(page, RECEIPT_FIELD_LABELS["transaction_date"], self.config.spatial)
            for m in matches:
                norm_d, is_ambig = normalize_date(m.value_text, self.config.normalization.default_date_order)
                if norm_d:
                    conf = calculate_extraction_confidence(
                        label_match_strength=0.95,
                        pattern_validity_score=0.90 if not is_ambig else 0.70,
                        spatial_score=m.spatial_score,
                        candidate_count=len(matches),
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    candidates.append(
                        ExtractionCandidate(
                            value=m.value_text,
                            raw_value=m.value_text,
                            normalized_value=norm_d,
                            confidence=conf,
                            source_text=m.source_line,
                            page_number=page.page_number,
                            bounding_box=m.value_bbox,
                            ocr_confidence=m.mean_ocr_confidence,
                            matched_label=m.label_text,
                        )
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        winner = candidates[0]
        prov = Provenance(
            document_id=document.id,
            page_number=winner.page_number,
            raw_text=winner.source_text,
            bounding_box=winner.bounding_box,
            ocr_confidence=winner.ocr_confidence,
        )
        return ExtractedField(
            name="transaction_date",
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.DATE,
            provenance=prov,
            extraction_confidence=winner.confidence,
            is_required=True,
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
        )

    def _extract_transaction_time(self, document: Document) -> Optional[ExtractedField]:
        for page in document.pages:
            matches = find_label_value_matches(page, RECEIPT_FIELD_LABELS["transaction_time"], self.config.spatial)
            if matches:
                m = matches[0]
                val = normalize_text(m.value_text)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name="transaction_time",
                    value=val,
                    normalized_value=val,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.88,
                    matched_label=m.label_text,
                )
            # Regex pattern fallback
            match = TIME_PATTERN.search(page.raw_text)
            if match:
                val = match.group(0)
                box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=val,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                    extraction_method=ExtractionMethod.REGEX_PATTERN,
                )
                return ExtractedField(
                    name="transaction_time",
                    value=val,
                    normalized_value=val,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.80,
                )
        return None

    def _extract_money_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
    ) -> Optional[ExtractedField]:
        candidates: List[ExtractionCandidate] = []
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                amt, curr = normalize_money(m.value_text)
                if amt is not None:
                    conf = calculate_extraction_confidence(
                        label_match_strength=1.0,
                        pattern_validity_score=0.95,
                        spatial_score=m.spatial_score,
                        candidate_count=len(matches),
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    candidates.append(
                        ExtractionCandidate(
                            value=m.value_text,
                            raw_value=m.value_text,
                            normalized_value=amt,
                            confidence=conf,
                            source_text=m.source_line,
                            page_number=page.page_number,
                            bounding_box=m.value_bbox,
                            ocr_confidence=m.mean_ocr_confidence,
                            matched_label=m.label_text,
                            details={"currency": curr},
                        )
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        winner = candidates[0]
        prov = Provenance(
            document_id=document.id,
            page_number=winner.page_number,
            raw_text=winner.source_text,
            bounding_box=winner.bounding_box,
            ocr_confidence=winner.ocr_confidence,
        )
        return ExtractedField(
            name=field_name,
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.CURRENCY,
            provenance=prov,
            extraction_confidence=winner.confidence,
            is_required=(field_name == "total"),
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
            metadata={"currency": winner.details.get("currency")},
        )

    def _extract_payment_method(self, document: Document) -> Optional[ExtractedField]:
        # Check explicit methods: Cash, Visa, Mastercard, Amex, Debit, Credit
        for page in document.pages:
            text_lower = page.raw_text.lower()
            for method in ("cash", "visa", "mastercard", "amex", "debit card", "credit card", "apple pay"):
                if re.search(rf"\b{re.escape(method)}\b", text_lower):
                    box, ocr_c = locate_value_in_regions(method, page.ocr_text_regions)
                    prov = Provenance(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=method.title(),
                        bounding_box=box,
                        ocr_confidence=ocr_c,
                        extraction_method=ExtractionMethod.KEY_VALUE_HEURISTIC,
                    )
                    return ExtractedField(
                        name="payment_method",
                        value=method.upper(),
                        normalized_value=method.upper(),
                        field_type=FieldType.STRING,
                        provenance=prov,
                        extraction_confidence=0.90,
                    )
        return None

    def _infer_currency(
        self,
        extracted_fields: Dict[str, ExtractedField],
        document: Document,
    ) -> Optional[ExtractedField]:
        for fname in ("total", "subtotal", "tax"):
            f = extracted_fields.get(fname)
            if f and f.metadata.get("currency"):
                curr = f.metadata["currency"]
                prov = Provenance(
                    document_id=document.id,
                    page_number=f.page_number,
                    raw_text=f.source_text,
                    bounding_box=f.bounding_box,
                    ocr_confidence=f.source_ocr_confidence,
                )
                return ExtractedField(
                    name="currency",
                    value=curr,
                    normalized_value=curr,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.92,
                )
        return None

    def _extract_text_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
        field_type: FieldType = FieldType.STRING,
    ) -> Optional[ExtractedField]:
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            if matches:
                m = matches[0]
                val = normalize_text(m.value_text)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name=field_name,
                    value=val,
                    normalized_value=val,
                    field_type=field_type,
                    provenance=prov,
                    extraction_confidence=0.85,
                    matched_label=m.label_text,
                )
        return None
