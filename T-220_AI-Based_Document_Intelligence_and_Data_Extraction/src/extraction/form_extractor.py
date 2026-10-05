"""Structured field extractor for Application and Intake Form documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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
    normalize_email,
    normalize_identifier,
    normalize_phone,
    normalize_text,
)
from src.extraction.patterns import FORM_FIELD_LABELS
from src.extraction.spatial import (
    find_label_value_matches,
    locate_value_in_regions,
)


class FormExtractor:
    """Extracts labeled key-value fields and personal data from structured forms."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    def extract(self, document: Document) -> Dict[str, ExtractedField]:
        """Extract generic form fields across all pages of a document."""
        extracted_fields: Dict[str, ExtractedField] = {}

        # 1. Applicant Name / Full Name
        name_f = self._extract_text_field(document, "applicant_name", FORM_FIELD_LABELS["applicant_name"])
        if name_f:
            extracted_fields["applicant_name"] = name_f

        # 2. First Name & Last Name
        fname_f = self._extract_text_field(document, "first_name", FORM_FIELD_LABELS["first_name"])
        if fname_f:
            extracted_fields["first_name"] = fname_f

        lname_f = self._extract_text_field(document, "last_name", FORM_FIELD_LABELS["last_name"])
        if lname_f:
            extracted_fields["last_name"] = lname_f

        # 3. Date of Birth
        dob_f = self._extract_date_field(document, "date_of_birth", FORM_FIELD_LABELS["date_of_birth"])
        if dob_f:
            extracted_fields["date_of_birth"] = dob_f

        # 4. Phone Number
        phone_f = self._extract_phone_field(document, "phone", FORM_FIELD_LABELS["phone"])
        if phone_f:
            extracted_fields["phone"] = phone_f

        # 5. Email
        email_f = self._extract_email_field(document, "email", FORM_FIELD_LABELS["email"])
        if email_f:
            extracted_fields["email"] = email_f

        # 6. Address
        addr_f = self._extract_text_field(document, "address", FORM_FIELD_LABELS["address"], FieldType.ADDRESS)
        if addr_f:
            extracted_fields["address"] = addr_f

        # 7. Form Number / Reference
        form_num_f = self._extract_text_field(document, "form_number", FORM_FIELD_LABELS["form_number"], FieldType.IDENTIFIER)
        if form_num_f:
            extracted_fields["form_number"] = form_num_f

        # 8. Signature Indicator
        sig_f = self._extract_signature_field(document)
        if sig_f:
            extracted_fields["signature_indicator"] = sig_f

        return extracted_fields

    def _extract_text_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
        field_type: FieldType = FieldType.STRING,
    ) -> Optional[ExtractedField]:
        candidates: List[ExtractionCandidate] = []
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                norm_val = normalize_text(m.value_text)
                # Ignore blank placeholders like "_________"
                clean_val = re.sub(r"^[_\s.\-]+|[_\s.\-]+$", "", norm_val)
                if clean_val and len(clean_val) >= 2:
                    # Semantic validation checks
                    if field_name == "address":
                        if "@" in clean_val or re.search(r"(?i)\b(?:email|e-mail|web|ip|mac|url)\s+address\b", m.source_line or ""):
                            continue
                    elif field_name == "email":
                        if "@" not in clean_val:
                            continue

                    conf = calculate_extraction_confidence(
                        label_match_strength=1.0,
                        pattern_validity_score=0.90,
                        spatial_score=m.spatial_score,
                        candidate_count=len(matches),
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    candidates.append(
                        ExtractionCandidate(
                            value=clean_val,
                            raw_value=m.value_text,
                            normalized_value=clean_val,
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
            name=field_name,
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=field_type,
            provenance=prov,
            extraction_confidence=winner.confidence,
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
        )

    def _extract_date_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
    ) -> Optional[ExtractedField]:
        candidates: List[ExtractionCandidate] = []
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                norm_d, is_ambig = normalize_date(m.value_text, self.config.normalization.default_date_order)
                if norm_d:
                    conf = calculate_extraction_confidence(
                        label_match_strength=1.0,
                        pattern_validity_score=0.95 if not is_ambig else 0.70,
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
            name=field_name,
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.DATE,
            provenance=prov,
            extraction_confidence=winner.confidence,
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
        )

    def _extract_phone_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
    ) -> Optional[ExtractedField]:
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                norm_p = normalize_phone(m.value_text)
                if norm_p:
                    prov = Provenance(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=m.source_line,
                        bounding_box=m.value_bbox,
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    return ExtractedField(
                        name=field_name,
                        value=m.value_text,
                        normalized_value=norm_p,
                        field_type=FieldType.PHONE,
                        provenance=prov,
                        extraction_confidence=0.88,
                        matched_label=m.label_text,
                    )
        return None

    def _extract_email_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
    ) -> Optional[ExtractedField]:
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                norm_e = normalize_email(m.value_text)
                if norm_e:
                    prov = Provenance(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=m.source_line,
                        bounding_box=m.value_bbox,
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    return ExtractedField(
                        name=field_name,
                        value=m.value_text,
                        normalized_value=norm_e,
                        field_type=FieldType.EMAIL,
                        provenance=prov,
                        extraction_confidence=0.90,
                        matched_label=m.label_text,
                    )
        return None

    def _extract_signature_field(self, document: Document) -> Optional[ExtractedField]:
        for page in document.pages:
            matches = find_label_value_matches(page, FORM_FIELD_LABELS["signature_indicator"], self.config.spatial)
            if matches:
                m = matches[0]
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name="signature_indicator",
                    value="PRESENT",
                    normalized_value=True,
                    field_type=FieldType.BOOLEAN,
                    provenance=prov,
                    extraction_confidence=0.85,
                    matched_label=m.label_text,
                )
        return None
