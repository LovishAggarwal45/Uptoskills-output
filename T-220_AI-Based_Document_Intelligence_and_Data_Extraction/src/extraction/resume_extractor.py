"""Structured field extractor for Resumes, CVs, and Candidate Profiles."""

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
    normalize_email,
    normalize_phone,
    normalize_text,
)
from src.extraction.patterns import (
    EMAIL_PATTERN,
    GITHUB_PATTERN,
    LINKEDIN_PATTERN,
    PHONE_PATTERN,
    RESUME_FIELD_LABELS,
    SECTION_HEADER_PATTERN,
)
from src.extraction.spatial import (
    find_label_value_matches,
    locate_value_in_regions,
)


class ResumeExtractor:
    """Extracts candidate profile data, contact details, skills, and sections from resumes and CVs."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    def extract(self, document: Document) -> Dict[str, ExtractedField]:
        """Extract structured resume fields across document pages."""
        extracted_fields: Dict[str, ExtractedField] = {}

        if not document.pages:
            return extracted_fields

        first_page = document.pages[0]
        full_text = "\n".join(p.raw_text or "" for p in document.pages)

        # 1. Candidate Name
        name_f = self._extract_candidate_name(document)
        if name_f:
            extracted_fields["candidate_name"] = name_f

        # 2. Email Address
        email_f = self._extract_email(document)
        if email_f:
            extracted_fields["email"] = email_f

        # 3. Phone Number
        phone_f = self._extract_phone(document)
        if phone_f:
            extracted_fields["phone"] = phone_f

        # 4. LinkedIn Profile
        linkedin_f = self._extract_linkedin(document)
        if linkedin_f:
            extracted_fields["linkedin_url"] = linkedin_f

        # 5. GitHub Profile
        github_f = self._extract_github(document)
        if github_f:
            extracted_fields["github_url"] = github_f

        # 6. Location / Address
        loc_f = self._extract_location(document)
        if loc_f:
            extracted_fields["location"] = loc_f

        # 7. Summary / Objective Section
        sum_f = self._extract_section(document, ["summary", "objective", "profile"], "summary")
        if sum_f:
            extracted_fields["summary"] = sum_f

        # 8. Skills Section
        skills_f = self._extract_section(document, ["skills", "technical skills", "competencies"], "skills", FieldType.STRING)
        if skills_f:
            extracted_fields["skills"] = skills_f

        # 9. Education Section
        edu_f = self._extract_section(document, ["education", "academic"], "education")
        if edu_f:
            extracted_fields["education"] = edu_f

        # 10. Experience Section
        exp_f = self._extract_section(document, ["experience", "employment history", "work experience"], "experience")
        if exp_f:
            extracted_fields["experience"] = exp_f

        # 11. Certifications
        cert_f = self._extract_section(document, ["certifications", "certificates", "courses"], "certifications")
        if cert_f:
            extracted_fields["certifications"] = cert_f

        return extracted_fields

    def _extract_candidate_name(self, document: Document) -> Optional[ExtractedField]:
        """Identify candidate full name from header or explicit label."""
        first_page = document.pages[0]

        # 1. Check for explicit label
        matches = find_label_value_matches(first_page, RESUME_FIELD_LABELS["candidate_name"], self.config.spatial)
        for m in matches:
            clean = normalize_text(m.value_text)
            if clean and len(clean.split()) in (2, 3, 4) and not re.search(r"[@\d]", clean):
                prov = Provenance(
                    document_id=document.id,
                    page_number=1,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name="candidate_name",
                    value=clean,
                    normalized_value=clean,
                    field_type=FieldType.STRING,
                    provenance=prov,
                    extraction_confidence=0.92,
                    matched_label=m.label_text,
                )

        # 2. Check top lines of Page 1
        lines = [l.strip() for l in (first_page.raw_text or "").split("\n") if l.strip()]
        for idx, line in enumerate(lines[:6]):
            # Skip emails, phones, URLs, and section headers
            if "@" in line or "http" in line.lower() or "github" in line.lower() or "linkedin" in line.lower():
                continue
            if SECTION_HEADER_PATTERN.match(line):
                continue
            if re.search(r"\b(?:curriculum\s+vitae|resume)\b", line, re.I):
                continue

            words = line.split()
            if 2 <= len(words) <= 5 and all(w[0].isupper() or w in ("de", "van", "von", "la") for w in words if w.isalpha()):
                if not re.search(r"[\d@#$%^&*()_+={}\[\]|\\<>]", line):
                    box, ocr_c = locate_value_in_regions(line, first_page.ocr_text_regions)
                    prov = Provenance(
                        document_id=document.id,
                        page_number=1,
                        raw_text=line,
                        bounding_box=box,
                        ocr_confidence=ocr_c,
                    )
                    return ExtractedField(
                        name="candidate_name",
                        value=line,
                        normalized_value=line,
                        field_type=FieldType.STRING,
                        provenance=prov,
                        extraction_confidence=0.88,
                    )

        return None

    def _extract_email(self, document: Document) -> Optional[ExtractedField]:
        """Extract primary candidate email address with spatial provenance."""
        for page in document.pages:
            text = page.raw_text or ""
            match = EMAIL_PATTERN.search(text)
            if match:
                val = match.group(0)
                norm_val = normalize_email(val)
                box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=val,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                )
                return ExtractedField(
                    name="email",
                    value=val,
                    normalized_value=norm_val,
                    field_type=FieldType.EMAIL,
                    provenance=prov,
                    extraction_confidence=0.95,
                )
        return None

    def _extract_phone(self, document: Document) -> Optional[ExtractedField]:
        """Extract primary contact phone number."""
        for page in document.pages:
            text = page.raw_text or ""
            match = PHONE_PATTERN.search(text)
            if match:
                val = match.group(0)
                norm_val = normalize_phone(val)
                if norm_val and len(norm_val) >= 7:
                    box, ocr_c = locate_value_in_regions(val, page.ocr_text_regions)
                    prov = Provenance(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=val,
                        bounding_box=box,
                        ocr_confidence=ocr_c,
                    )
                    return ExtractedField(
                        name="phone",
                        value=val,
                        normalized_value=norm_val,
                        field_type=FieldType.PHONE,
                        provenance=prov,
                        extraction_confidence=0.90,
                    )
        return None

    def _extract_linkedin(self, document: Document) -> Optional[ExtractedField]:
        """Extract LinkedIn profile link or username."""
        for page in document.pages:
            text = page.raw_text or ""
            match = LINKEDIN_PATTERN.search(text)
            if match:
                full_val = match.group(0)
                username = match.group(1)
                norm_url = f"https://linkedin.com/in/{username}"
                box, ocr_c = locate_value_in_regions(full_val, page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=full_val,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                )
                return ExtractedField(
                    name="linkedin_url",
                    value=full_val,
                    normalized_value=norm_url,
                    field_type=FieldType.URL,
                    provenance=prov,
                    extraction_confidence=0.92,
                )
        return None

    def _extract_github(self, document: Document) -> Optional[ExtractedField]:
        """Extract GitHub profile link or username."""
        for page in document.pages:
            text = page.raw_text or ""
            match = GITHUB_PATTERN.search(text)
            if match:
                full_val = match.group(0)
                username = match.group(1)
                norm_url = f"https://github.com/{username}"
                box, ocr_c = locate_value_in_regions(full_val, page.ocr_text_regions)
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=full_val,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                )
                return ExtractedField(
                    name="github_url",
                    value=full_val,
                    normalized_value=norm_url,
                    field_type=FieldType.URL,
                    provenance=prov,
                    extraction_confidence=0.92,
                )
        return None

    def _extract_location(self, document: Document) -> Optional[ExtractedField]:
        """Extract candidate location/city/address."""
        first_page = document.pages[0]
        matches = find_label_value_matches(first_page, RESUME_FIELD_LABELS["location"], self.config.spatial)
        for m in matches:
            clean = normalize_text(m.value_text)
            if clean and len(clean) >= 3 and not re.search(r"[@\d]{6,}", clean):
                prov = Provenance(
                    document_id=document.id,
                    page_number=1,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name="location",
                    value=clean,
                    normalized_value=clean,
                    field_type=FieldType.ADDRESS,
                    provenance=prov,
                    extraction_confidence=0.85,
                    matched_label=m.label_text,
                )
        return None

    def _extract_section(
        self,
        document: Document,
        section_keywords: List[str],
        field_name: str,
        field_type: FieldType = FieldType.STRING,
    ) -> Optional[ExtractedField]:
        """Extract body text belonging to a named section heading."""
        for page in document.pages:
            lines = (page.raw_text or "").split("\n")
            in_section = False
            section_lines: List[str] = []

            for line in lines:
                clean = line.strip()
                if not clean:
                    continue

                words = clean.split()
                clean_lower = clean.lower().rstrip(":-")

                # Check if this line is a section header (short line matching keyword)
                is_header = False
                if len(words) <= 4:
                    for kw in section_keywords:
                        if clean_lower == kw or clean_lower.startswith(kw + " ") or clean_lower.endswith(" " + kw):
                            is_header = True
                            break

                if is_header:
                    in_section = True
                    continue

                # If in section and encounter another recognized section header, terminate section
                if in_section:
                    if (
                        SECTION_HEADER_PATTERN.match(clean)
                        or (clean.isupper() and len(words) <= 4 and any(kw in clean for kw in ["EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS", "SUMMARY", "WORK", "CERTIFICATIONS", "AWARDS"]))
                    ):
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
                    field_type=field_type,
                    provenance=prov,
                    extraction_confidence=0.85,
                )
        return None


__all__ = ["ResumeExtractor"]
