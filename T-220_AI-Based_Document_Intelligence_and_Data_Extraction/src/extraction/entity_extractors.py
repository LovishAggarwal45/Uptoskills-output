"""Generic named and typed entity extractors for universal document processing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.config import ExtractionConfig
from src.core.models import Document, DocumentPage
from src.core.types import ExtractionMethod
from src.extraction.confidence import calculate_extraction_confidence
from src.extraction.models import EntityType, ExtractedEntity
from src.extraction.normalizers import (
    normalize_date,
    normalize_email,
    normalize_identifier,
    normalize_money,
    normalize_phone,
    normalize_text,
)
from src.extraction.patterns import (
    EMAIL_PATTERN,
    GENERIC_ID_PATTERN,
    MONEY_PATTERN,
    ORG_SUFFIX_PATTERN,
    PHONE_PATTERN,
    TAX_ID_PATTERN,
    URL_PATTERN,
)
from src.extraction.spatial import locate_value_in_regions

ADDRESS_KEYWORD_PATTERN = re.compile(
    r"\b(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Suite|Ste\.?|Floor|Fl\.?|Building|Bldg\.?|P\.?O\.?\s*Box)\b|\b\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)


class GenericEntityExtractor:
    """Extracts universal entities (dates, money, emails, phones, URLs, persons, IDs, organizations, addresses)."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    def extract_entities(self, document: Document) -> List[ExtractedEntity]:
        """Extract generic entities across all pages of a document."""
        entities: List[ExtractedEntity] = []
        cfg_ent = self.config.entities

        for page in document.pages:
            page_text = page.raw_text or ""
            if not page_text.strip():
                continue

            seen_spans: Set[Tuple[int, int, str]] = set()

            # 1. Emails
            if cfg_ent.extract_email:
                for match in EMAIL_PATTERN.finditer(page_text):
                    span = (match.start(), match.end())
                    val = match.group(0)
                    norm_val = normalize_email(val)
                    if norm_val:
                        box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.90,
                            pattern_validity_score=1.0,
                            spatial_score=1.0,
                            ocr_confidence=ocr_c,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type=EntityType.EMAIL,
                                value=val,
                                normalized_value=norm_val,
                                confidence=conf,
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                source_text=val,
                                character_span=span,
                                extraction_method=ExtractionMethod.REGEX_PATTERN,
                            )
                        )
                        seen_spans.add((span[0], span[1], "email"))

            # 2. Phone Numbers
            if cfg_ent.extract_phone:
                for match in PHONE_PATTERN.finditer(page_text):
                    span = (match.start(), match.end())
                    val = match.group(0)
                    norm_val = normalize_phone(val)
                    if norm_val and len(norm_val) >= 7:
                        box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.85,
                            pattern_validity_score=0.95,
                            spatial_score=1.0,
                            ocr_confidence=ocr_c,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type=EntityType.PHONE,
                                value=val,
                                normalized_value=norm_val,
                                confidence=conf,
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                source_text=val,
                                character_span=span,
                                extraction_method=ExtractionMethod.REGEX_PATTERN,
                            )
                        )
                        seen_spans.add((span[0], span[1], "phone"))

            # 3. URLs & Profile Links
            for match in URL_PATTERN.finditer(page_text):
                span = (match.start(), match.end())
                val = match.group(0).strip()
                if val and len(val) >= 4 and not val.startswith("@"):
                    box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                    conf = calculate_extraction_confidence(
                        label_match_strength=0.90,
                        pattern_validity_score=0.95,
                        spatial_score=1.0,
                        ocr_confidence=ocr_c,
                    )
                    entities.append(
                        ExtractedEntity(
                            entity_type=EntityType.URL,
                            value=val,
                            normalized_value=val,
                            confidence=conf,
                            page_number=page.page_number,
                            bounding_box=box,
                            ocr_confidence=ocr_c,
                            source_text=val,
                            character_span=span,
                            extraction_method=ExtractionMethod.REGEX_PATTERN,
                        )
                    )
                    seen_spans.add((span[0], span[1], "url"))

            # 4. Dates
            if cfg_ent.extract_dates:
                date_candidates = re.finditer(
                    r"\b(?:\d{4}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])|"
                    r"(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:\d{4}|\d{2})|"
                    r"(?:0?[1-9]|[12]\d|3[01])\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,\s]+\d{4})\b",
                    page_text,
                    re.IGNORECASE,
                )
                for match in date_candidates:
                    span = (match.start(), match.end())
                    val = match.group(0)
                    norm_val, is_ambig = normalize_date(
                        val,
                        default_order=self.config.normalization.default_date_order,
                    )
                    if norm_val:
                        box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.90,
                            pattern_validity_score=0.95 if not is_ambig else 0.70,
                            spatial_score=1.0,
                            ocr_confidence=ocr_c,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type=EntityType.DATE,
                                value=val,
                                normalized_value=norm_val,
                                confidence=conf,
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                source_text=val,
                                character_span=span,
                                extraction_method=ExtractionMethod.REGEX_PATTERN,
                                metadata={"is_ambiguous": is_ambig},
                            )
                        )
                        seen_spans.add((span[0], span[1], "date"))

            # 5. Monetary Values
            if cfg_ent.extract_money:
                for match in MONEY_PATTERN.finditer(page_text):
                    span = (match.start(), match.end())
                    val = match.group(0)
                    amt, curr = normalize_money(val)
                    if amt is not None:
                        # Avoid bare integers without currency symbols or decimals (e.g. street numbers or zip codes)
                        if not curr and "." not in val and "," not in val:
                            continue
                        box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.85 if curr else 0.70,
                            pattern_validity_score=0.95,
                            spatial_score=1.0,
                            ocr_confidence=ocr_c,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type=EntityType.MONEY,
                                value=val,
                                normalized_value={"amount": amt, "currency": curr},
                                confidence=conf,
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                source_text=val,
                                character_span=span,
                                extraction_method=ExtractionMethod.REGEX_PATTERN,
                            )
                        )
                        seen_spans.add((span[0], span[1], "money"))

            # 6. Identifiers & Tax IDs
            if cfg_ent.extract_identifiers:
                for pat in (GENERIC_ID_PATTERN, TAX_ID_PATTERN):
                    for match in pat.finditer(page_text):
                        span = (match.start(), match.end())
                        val = match.group(0)
                        norm_val = normalize_identifier(val)
                        if norm_val and len(norm_val) >= 4:
                            box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                            conf = calculate_extraction_confidence(
                                label_match_strength=0.80,
                                pattern_validity_score=0.90,
                                spatial_score=1.0,
                                ocr_confidence=ocr_c,
                            )
                            entities.append(
                                ExtractedEntity(
                                    entity_type=EntityType.IDENTIFIER,
                                    value=val,
                                    normalized_value=norm_val,
                                    confidence=conf,
                                    page_number=page.page_number,
                                    bounding_box=box,
                                    ocr_confidence=ocr_c,
                                    source_text=val,
                                    character_span=span,
                                    extraction_method=ExtractionMethod.REGEX_PATTERN,
                                )
                            )

            # 7. Organizations (Company names with corporate identifiers)
            if cfg_ent.extract_organizations:
                for line in page_text.split("\n"):
                    line_clean = normalize_text(line)
                    if ORG_SUFFIX_PATTERN.search(line_clean) and len(line_clean) < 60:
                        box, ocr_c = locate_value_in_regions(line_clean, page.ocr_text_regions)
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.85,
                            pattern_validity_score=0.85,
                            spatial_score=1.0,
                            ocr_confidence=ocr_c,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type=EntityType.ORGANIZATION,
                                value=line_clean,
                                normalized_value=line_clean,
                                confidence=conf,
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                source_text=line_clean,
                                extraction_method=ExtractionMethod.REGEX_PATTERN,
                            )
                        )

            # 8. Addresses (Lines matching address markers)
            if cfg_ent.extract_addresses:
                for line in page_text.split("\n"):
                    line_clean = normalize_text(line)
                    if ADDRESS_KEYWORD_PATTERN.search(line_clean) and 10 < len(line_clean) < 100:
                        box, ocr_c = locate_value_in_regions(line_clean, page.ocr_text_regions)
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.80,
                            pattern_validity_score=0.80,
                            spatial_score=1.0,
                            ocr_confidence=ocr_c,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type=EntityType.ADDRESS,
                                value=line_clean,
                                normalized_value=line_clean,
                                confidence=conf,
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                source_text=line_clean,
                                extraction_method=ExtractionMethod.KEY_VALUE_HEURISTIC,
                            )
                        )

            # 9. Persons (Labeled names)
            for line in page_text.split("\n"):
                line_clean = normalize_text(line)
                m_person = re.match(r"(?i)^(?:Candidate|Applicant|Author|Student|Patient|Employee|Name)\s*:\s*([A-Za-z\s.\-]{2,40})$", line_clean)
                if m_person:
                    pname = m_person.group(1).strip()
                    box, ocr_c = locate_value_in_regions(pname, page.ocr_text_regions)
                    conf = calculate_extraction_confidence(
                        label_match_strength=0.95,
                        pattern_validity_score=0.90,
                        spatial_score=1.0,
                        ocr_confidence=ocr_c,
                    )
                    entities.append(
                        ExtractedEntity(
                            entity_type=EntityType.PERSON,
                            value=pname,
                            normalized_value=pname,
                            confidence=conf,
                            page_number=page.page_number,
                            bounding_box=box,
                            ocr_confidence=ocr_c,
                            source_text=line_clean,
                            extraction_method=ExtractionMethod.KEY_VALUE_HEURISTIC,
                        )
                    )

        return entities
