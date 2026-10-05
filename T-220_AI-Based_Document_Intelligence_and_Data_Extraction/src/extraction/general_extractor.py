"""Structured field extractor for General Documents, Reports, Memos, and Unknown formats."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.config import ExtractionConfig
from src.core.models import Document, ExtractedField, Provenance
from src.core.types import (
    ExtractionMethod,
    FieldType,
)
from src.extraction.confidence import calculate_extraction_confidence
from src.extraction.normalizers import (
    normalize_date,
    normalize_text,
)
from src.extraction.patterns import (
    GENERAL_FIELD_LABELS,
    SECTION_HEADER_PATTERN,
)
from src.extraction.spatial import (
    find_label_value_matches,
    locate_value_in_regions,
)


class GeneralDocumentExtractor:
    """Extracts metadata, generic key-value fields, and structural sections from general/unknown documents."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    def extract(self, document: Document) -> Dict[str, ExtractedField]:
        """Extract generic fields and labeled key-values across all pages."""
        extracted_fields: Dict[str, ExtractedField] = {}

        if not document.pages:
            return extracted_fields

        first_page = document.pages[0]

        # 1. Document Title / Subject
        title_f = self._extract_title(document)
        if title_f:
            extracted_fields["document_title"] = title_f

        # 2. Author / Issuer
        author_f = self._extract_field(document, "author", GENERAL_FIELD_LABELS["author"])
        if author_f:
            extracted_fields["author"] = author_f

        # 3. Recipient
        recip_f = self._extract_field(document, "recipient", GENERAL_FIELD_LABELS["recipient"])
        if recip_f:
            extracted_fields["recipient"] = recip_f

        # 4. Primary Date
        date_f = self._extract_date_field(document, "document_date", GENERAL_FIELD_LABELS["date"])
        if date_f:
            extracted_fields["document_date"] = date_f

        # 5. Organization
        org_f = self._extract_field(document, "organization", GENERAL_FIELD_LABELS["organization"], FieldType.ORGANIZATION)
        if org_f:
            extracted_fields["organization"] = org_f

        # 6. Executive Summary / Abstract
        sum_f = self._extract_section(document, ["executive summary", "abstract", "overview", "summary"], "summary")
        if sum_f:
            extracted_fields["summary"] = sum_f

        # 7. Generic Key-Value Pairs across pages
        kv_fields = self._extract_generic_key_values(document)
        for k, v in kv_fields.items():
            if k not in extracted_fields:
                extracted_fields[k] = v

        return extracted_fields

    def _extract_title(self, document: Document) -> Optional[ExtractedField]:
        """Extract document title from labeled subject or top header."""
        first_page = document.pages[0]

        # 1. Labeled title
        matches = find_label_value_matches(first_page, GENERAL_FIELD_LABELS["title"], self.config.spatial)
        for m in matches:
            clean = normalize_text(m.value_text)
            if clean and len(clean) >= 3:
                prov = Provenance(
                    document_id=document.id,
                    page_number=1,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name="document_title",
                    value=clean,
                    normalized_value=clean,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.90,
                    matched_label=m.label_text,
                )

        # 2. Top header line
        lines = [l.strip() for l in (first_page.raw_text or "").split("\n") if l.strip()]
        for line in lines[:3]:
            if len(line) >= 4 and not re.search(r"[@\d]{5,}", line) and not line.startswith("http"):
                box, ocr_c = locate_value_in_regions(line, first_page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=1,
                    raw_text=line,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                )
                return ExtractedField(
                    name="document_title",
                    value=line,
                    normalized_value=line,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.80,
                )

        return None

    def _extract_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
        field_type: FieldType = FieldType.STRING,
    ) -> Optional[ExtractedField]:
        """Extract labeled string field."""
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                clean = normalize_text(m.value_text)
                if clean and len(clean) >= 2:
                    prov = Provenance(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=m.source_line,
                        bounding_box=m.value_bbox,
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    return ExtractedField(
                        name=field_name,
                        value=clean,
                        normalized_value=clean,
                        field_type=field_type,
                        provenance=prov,
                        extraction_confidence=0.88,
                        matched_label=m.label_text,
                    )
        return None

    def _extract_date_field(
        self,
        document: Document,
        field_name: str,
        labels: List[str],
    ) -> Optional[ExtractedField]:
        """Extract labeled date field."""
        for page in document.pages:
            matches = find_label_value_matches(page, labels, self.config.spatial)
            for m in matches:
                norm_d, is_ambig = normalize_date(m.value_text, self.config.normalization.default_date_order)
                if norm_d:
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
                        normalized_value=norm_d,
                        field_type=FieldType.DATE,
                        provenance=prov,
                        extraction_confidence=0.90 if not is_ambig else 0.75,
                        matched_label=m.label_text,
                    )
        return None

    def _extract_section(
        self,
        document: Document,
        section_keywords: List[str],
        field_name: str,
    ) -> Optional[ExtractedField]:
        """Extract text content under matching section header."""
        for page in document.pages:
            lines = (page.raw_text or "").split("\n")
            in_section = False
            section_lines: List[str] = []

            for line in lines:
                clean = line.strip()
                if not clean:
                    continue

                is_header = any(clean.lower().startswith(kw) for kw in section_keywords)
                if is_header:
                    in_section = True
                    continue

                if in_section:
                    if SECTION_HEADER_PATTERN.match(clean) or (clean.isupper() and len(clean.split()) <= 4):
                        break
                    section_lines.append(clean)

            if section_lines:
                extracted_block = "\n".join(section_lines)
                first_line = section_lines[0]
                box, ocr_c = locate_value_in_regions(first_line, page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=first_line,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                )
                return ExtractedField(
                    name=field_name,
                    value=extracted_block,
                    normalized_value=extracted_block,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.85,
                )
        return None

    def _extract_generic_key_values(self, document: Document) -> Dict[str, ExtractedField]:
        """Extract general 'Key: Value' patterns from document text lines."""
        results: Dict[str, ExtractedField] = {}

        for page in document.pages:
            lines = (page.raw_text or "").split("\n")
            for line in lines:
                clean = line.strip()
                if ":" not in clean or clean.startswith("http"):
                    continue

                parts = clean.split(":", 1)
                k_raw = parts[0].strip()
                v_raw = parts[1].strip()

                # Key validation: 1 to 4 words, alphabetic, not excessively long
                k_words = k_raw.split()
                if 1 <= len(k_words) <= 4 and len(k_raw) < 30 and all(w.replace("-", "").replace("_", "").isalpha() for w in k_words):
                    clean_k = "_".join(w.lower() for w in k_words)
                    if v_raw and len(v_raw) >= 1:
                        box, ocr_c = locate_value_in_regions(v_raw, page.ocr_text_regions)
                        prov = Provenance(
                            document_id=document.id,
                            page_number=page.page_number,
                            raw_text=clean,
                            bounding_box=box,
                            ocr_confidence=ocr_c,
                        )
                        results[clean_k] = ExtractedField(
                            name=clean_k,
                            value=v_raw,
                            normalized_value=v_raw,
                            field_type=FieldType.STRING,
                            provenance=prov,
                            extraction_confidence=0.82,
                            matched_label=k_raw,
                        )

        return results


__all__ = ["GeneralDocumentExtractor"]
