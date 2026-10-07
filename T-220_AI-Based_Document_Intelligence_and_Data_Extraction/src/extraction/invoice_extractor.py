"""Structured field extractor for Invoice documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.config import ExtractionConfig
from src.core.models import BoundingBox, Document, DocumentPage, ExtractedField, Provenance
from src.core.types import (
    ConfidenceSource,
    ExtractionMethod,
    FieldType,
    ValidationStatus,
)
from src.extraction.confidence import calculate_extraction_confidence
from src.extraction.models import ExtractionCandidate
from src.extraction.normalizers import (
    isolate_tax_identifier,
    normalize_date,
    normalize_identifier,
    normalize_money,
    normalize_text,
)
from src.extraction.patterns import (
    INVOICE_FIELD_LABELS,
    INVOICE_ID_PATTERN,
    ORG_SUFFIX_PATTERN,
    TAX_ID_PATTERN,
)
from src.extraction.spatial import (
    SpatialMatch,
    find_label_value_matches,
    locate_value_in_regions,
)


GENERIC_PARTY_EXCLUSIONS: Set[str] = {
    "invoice", "tax invoice", "commercial invoice", "proforma invoice", "bill",
    "invoice date", "invoice date.", "date", "due date", "bill date", "issue date", "expiry date",
    "invoice number", "invoice no", "invoice #", "inv #", "inv no", "bill no",
    "bill to", "bill to:", "billed to", "billed to:", "invoice to", "invoice to:",
    "ship to", "ship to:", "shipped to", "shipped to:", "sold to", "sold to:",
    "vendor", "vendor:", "seller", "seller:", "supplier", "supplier:", "billed by", "billed by:",
    "customer", "customer:", "client", "client:", "recipient", "recipient:", "buyer", "buyer:",
    "purchase order", "po number", "po no", "po #", "p.o. #", "p.o. no.",
    "tax id", "vat no", "vat number", "gstin", "gst no", "ein", "tin",
    "subtotal", "tax", "total", "amount due", "balance due", "grand total",
    "page", "page 1", "page 1 of 1", "payment terms", "terms", "notes",
    "electronic reservation slip", "booked from", "booked to", "passenger details",
}


def is_invalid_party_name(text: Optional[str]) -> bool:
    """Check if candidate text is a generic label, date, monetary amount, or invalid party name."""
    if not text:
        return True

    cleaned = normalize_text(text).strip()
    if len(cleaned) < 2:
        return True

    lower = cleaned.lower().rstrip(":.#- ")

    if lower in GENERIC_PARTY_EXCLUSIONS:
        return True

    if re.match(
        r"^(?:invoice|inv|date|due\s*date|bill\s*to|billed\s*to|ship\s*to|shipped\s*to|sold\s*to|vendor|seller|customer|client|tax\s*id|vat|gstin|subtotal|total|amount|po\s*#|purchase\s*order)\b",
        lower,
    ):
        return True

    if re.match(
        r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$",
        cleaned,
    ):
        return True

    if re.match(r"^[$€£¥₹]?\s*[\d,]+(?:\.\d{2})?\s*[$€£¥₹]?$", cleaned):
        return True

    if "@" in cleaned and "." in cleaned:
        return True

    if re.match(r"^(https?://|www\.)", lower):
        return True

    if re.match(r"^\+?[\d\s().-]{7,}$", cleaned):
        return True

    if not re.search(r"[A-Za-z]", cleaned):
        return True

    return False


def clean_vendor_name(text: Optional[str]) -> str:
    """Remove trailing invoice document titles from vendor names."""
    cleaned = normalize_text(text).strip()

    # A document title alone is not a valid vendor name.
    if cleaned.upper().strip(" .:-") in {"INVOICE", "TAX INVOICE"}:
        return ""

    # Remove only a trailing document title, not occurrences inside a name.
    cleaned = re.sub(
        r"\s+(?:TAX\s+)?INVOICE\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.strip(" -:|")

    # Do not return a document title if nothing remains.
    if cleaned.upper().strip(" .:-") in {"INVOICE", "TAX INVOICE"}:
        return ""

    return cleaned


class InvoiceExtractor:
    """Extracts canonical invoice fields with layout-awareness and provenance tracking."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    def extract(self, document: Document) -> Dict[str, ExtractedField]:
        """Extract invoice-specific fields from all document pages."""
        extracted_fields: Dict[str, ExtractedField] = {}

        # 1. Invoice Number
        inv_num_field = self._extract_invoice_number(document)
        if inv_num_field:
            extracted_fields["invoice_number"] = inv_num_field

        # 2. Invoice Date
        inv_date_field = self._extract_date_field(
            document, "invoice_date", INVOICE_FIELD_LABELS["invoice_date"]
        )
        if inv_date_field:
            extracted_fields["invoice_date"] = inv_date_field

        # 3. Due Date
        due_date_field = self._extract_date_field(
            document, "due_date", INVOICE_FIELD_LABELS["due_date"]
        )
        if due_date_field:
            extracted_fields["due_date"] = due_date_field

        # 4. Vendor Name & Address
        vendor_name, vendor_addr = self._extract_vendor_info(document)
        if vendor_name:
            extracted_fields["vendor_name"] = vendor_name
        if vendor_addr:
            extracted_fields["vendor_address"] = vendor_addr

        # 5. Customer Name & Address (Bill To)
        cust_name, cust_addr = self._extract_customer_info(document)
        if cust_name:
            extracted_fields["customer_name"] = cust_name
        if cust_addr:
            extracted_fields["customer_address"] = cust_addr

        # 6. Monetary Fields: Subtotal, Tax, Total
        subtotal_field = self._extract_money_field(
            document, "subtotal", INVOICE_FIELD_LABELS["subtotal"]
        )
        if subtotal_field:
            extracted_fields["subtotal"] = subtotal_field

        tax_field = self._extract_money_field(
            document, "tax", INVOICE_FIELD_LABELS["tax"]
        )
        if tax_field:
            extracted_fields["tax"] = tax_field

        total_field = self._extract_money_field(
            document, "total", INVOICE_FIELD_LABELS["total"]
        )
        if total_field:
            extracted_fields["total"] = total_field

        # 7. Currency
        currency_field = self._infer_currency(extracted_fields, document)
        if currency_field:
            extracted_fields["currency"] = currency_field

        # 8. Payment Terms
        terms_field = self._extract_text_field(
            document, "payment_terms", INVOICE_FIELD_LABELS["payment_terms"]
        )
        
        if terms_field:
            # Keep only the actual payment terms, excluding sample disclaimers.
            value = str(
                terms_field.normalized_value or terms_field.value or ""
            ).strip()

            value = re.split(
                r"\s*(?:This is a fictional sample invoice|"
                r"This document is for testing)\b",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

            terms_field.value = value
            terms_field.normalized_value = value
            extracted_fields["payment_terms"] = terms_field

            disclaimer_markers = (
                "This is a fictional sample invoice",
                "This document is for testing",
            )

            for marker in disclaimer_markers:
                index = value.lower().find(marker.lower())
                if index >= 0:
                    value = value[:index].strip()
                    break

            terms_field.value = value
            terms_field.normalized_value = value
            extracted_fields["payment_terms"] = terms_field


        # 9. Purchase Order Number
        po_field = self._extract_text_field(
            document,
            "purchase_order_number",
            INVOICE_FIELD_LABELS["purchase_order_number"],
            FieldType.IDENTIFIER,
        )
        if po_field:
            extracted_fields["purchase_order_number"] = po_field

        # 10. Tax ID / VAT
        tax_id_field = self._extract_tax_id(document)
        if tax_id_field:
            extracted_fields["tax_id"] = tax_id_field

        return extracted_fields

    def _extract_invoice_number(
        self, document: Document
    ) -> Optional[ExtractedField]:
        candidates: List[ExtractionCandidate] = []

        for page in document.pages:
            matches = find_label_value_matches(
                page, INVOICE_FIELD_LABELS["invoice_number"], self.config.spatial
            )

            for m in matches:
                norm_val = normalize_identifier(m.value_text)
                if norm_val:
                    conf = calculate_extraction_confidence(
                        label_match_strength=1.0,
                        pattern_validity_score=(
                            0.95 if re.search(r"\d", norm_val) else 0.70
                        ),
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
                for match in INVOICE_ID_PATTERN.finditer(page.raw_text):
                    val = match.group(1) if match.groups() else match.group(0)
                    norm_val = normalize_identifier(val)

                    if norm_val:
                        box, ocr_c = locate_value_in_regions(
                            val, page.ocr_text_regions
                        )
                        conf = calculate_extraction_confidence(
                            label_match_strength=0.80,
                            pattern_validity_score=0.95,
                            spatial_score=0.70,
                            ocr_confidence=ocr_c,
                        )
                        candidates.append(
                            ExtractionCandidate(
                                value=norm_val,
                                raw_value=val,
                                normalized_value=norm_val,
                                confidence=conf,
                                source_text=match.group(0),
                                page_number=page.page_number,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                extraction_method=ExtractionMethod.REGEX_PATTERN,
                            )
                        )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        winner = candidates[0]
        is_ambig = (
            len(candidates) > 1
            and abs(candidates[0].confidence - candidates[1].confidence) < 0.10
        )

        prov = Provenance(
            document_id=document.id,
            page_number=winner.page_number,
            raw_text=winner.source_text or winner.raw_value,
            bounding_box=winner.bounding_box,
            ocr_confidence=winner.ocr_confidence,
            extraction_method=winner.extraction_method,
        )

        return ExtractedField(
            name="invoice_number",
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.IDENTIFIER,
            provenance=prov,
            extraction_confidence=winner.confidence,
            confidence_source=ConfidenceSource.RULE_SCORE,
            validation_status=ValidationStatus.UNVALIDATED,
            is_required=True,
            is_ambiguous=is_ambig,
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
                norm_date, is_ambig_date = normalize_date(
                    m.value_text,
                    default_order=self.config.normalization.default_date_order,
                )

                if norm_date:
                    conf = calculate_extraction_confidence(
                        label_match_strength=1.0,
                        pattern_validity_score=(
                            0.95 if not is_ambig_date else 0.70
                        ),
                        spatial_score=m.spatial_score,
                        candidate_count=len(matches),
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    candidates.append(
                        ExtractionCandidate(
                            value=m.value_text,
                            raw_value=m.value_text,
                            normalized_value=norm_date,
                            confidence=conf,
                            source_text=m.source_line,
                            page_number=page.page_number,
                            bounding_box=m.value_bbox,
                            ocr_confidence=m.mean_ocr_confidence,
                            matched_label=m.label_text,
                            details={"is_ambiguous_date": is_ambig_date},
                        )
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        winner = candidates[0]
        is_ambig = (
            len(candidates) > 1
            and abs(candidates[0].confidence - candidates[1].confidence) < 0.10
        )

        prov = Provenance(
            document_id=document.id,
            page_number=winner.page_number,
            raw_text=winner.source_text or winner.raw_value,
            bounding_box=winner.bounding_box,
            ocr_confidence=winner.ocr_confidence,
            extraction_method=winner.extraction_method,
        )

        return ExtractedField(
            name=field_name,
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.DATE,
            provenance=prov,
            extraction_confidence=winner.confidence,
            confidence_source=ConfidenceSource.RULE_SCORE,
            validation_status=ValidationStatus.UNVALIDATED,
            is_required=(field_name == "invoice_date"),
            is_ambiguous=(
                is_ambig or winner.details.get("is_ambiguous_date", False)
            ),
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
        )

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
        is_ambig = (
            len(candidates) > 1
            and abs(candidates[0].confidence - candidates[1].confidence) < 0.10
        )

        prov = Provenance(
            document_id=document.id,
            page_number=winner.page_number,
            raw_text=winner.source_text or winner.raw_value,
            bounding_box=winner.bounding_box,
            ocr_confidence=winner.ocr_confidence,
            extraction_method=winner.extraction_method,
        )

        return ExtractedField(
            name=field_name,
            value=winner.value,
            normalized_value=winner.normalized_value,
            field_type=FieldType.CURRENCY,
            provenance=prov,
            extraction_confidence=winner.confidence,
            confidence_source=ConfidenceSource.RULE_SCORE,
            validation_status=ValidationStatus.UNVALIDATED,
            is_required=(field_name == "total"),
            is_ambiguous=is_ambig,
            candidates=[c.to_dict() for c in candidates],
            matched_label=winner.matched_label,
            metadata={"currency": winner.details.get("currency")},
        )

    def _extract_vendor_info(
        self,
        document: Document,
    ) -> Tuple[Optional[ExtractedField], Optional[ExtractedField]]:
        if not document.pages:
            return None, None

        page = document.pages[0]

        document_text = "\n".join(
            filter(
                None,
                [
                    page.raw_text or "",
                    " ".join(region.text for region in page.ocr_text_regions),
                ],
            )
        ).lower()

        is_railway_ticket = (
            "electronic reservation slip" in document_text
            and ("indian railways" in document_text or "irctc" in document_text)
        )

        if is_railway_ticket:
            issuer = (
                "Indian Railways"
                if "indian railways" in document_text
                else "IRCTC"
            )
            provenance = Provenance(
                document_id=document.id,
                page_number=page.page_number,
                raw_text=issuer,
                bounding_box=None,
                ocr_confidence=None,
                extraction_method=ExtractionMethod.LAYOUT_ANALYSIS,
            )
            return ExtractedField(
                name="vendor_name",
                value=issuer,
                normalized_value=issuer,
                field_type=FieldType.ORGANIZATION,
                provenance=provenance,
                extraction_confidence=0.90,
                is_required=True,
            ), None

        # 1. Explicit label-based match.
        matches = find_label_value_matches(
            page,
            INVOICE_FIELD_LABELS["vendor_name"],
            self.config.spatial,
        )

        for m in matches:
            # Clean the extracted value but preserve the original OCR source.
            val = clean_vendor_name(m.value_text)

            if val and not is_invalid_party_name(val):
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=m.source_line,
                    bounding_box=m.value_bbox,
                    ocr_confidence=m.mean_ocr_confidence,
                )
                return ExtractedField(
                    name="vendor_name",
                    value=val,
                    normalized_value=val,
                    field_type=FieldType.ORGANIZATION,
                    provenance=prov,
                    extraction_confidence=0.90,
                    is_required=True,
                    matched_label=m.label_text,
                ), None

        # 2. Prominent header / organization analysis.
        lines = [
            normalize_text(line)
            for line in (page.raw_text or "").split("\n")
            if normalize_text(line)
        ]

        vendor_candidates: List[Tuple[str, float]] = []

        for line_idx, original_line in enumerate(lines[:8]):
            line = clean_vendor_name(original_line)

            if is_invalid_party_name(line):
                continue

            if re.search(
                r"\b(?:street|st\b|ave\b|avenue|blvd|boulevard|rd\b|road|lane|drive|pkwy|parkway|suite|ste\b|floor|fl\b|box\s+\d+|zip|pin\b)\b",
                line,
                re.IGNORECASE,
            ):
                continue

            if re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", line):
                continue

            score = 0.70 - (line_idx * 0.05)

            if ORG_SUFFIX_PATTERN.search(line):
                score += 0.25

            vendor_candidates.append((line, score))

        if vendor_candidates:
            vendor_candidates.sort(key=lambda c: c[1], reverse=True)
            winner_text, winner_score = vendor_candidates[0]
            box, ocr_c = locate_value_in_regions(
                winner_text, page.ocr_text_regions
            )

            prov = Provenance(
                document_id=document.id,
                page_number=page.page_number,
                raw_text=winner_text,
                bounding_box=box,
                ocr_confidence=ocr_c,
                extraction_method=ExtractionMethod.LAYOUT_ANALYSIS,
            )

            return ExtractedField(
                name="vendor_name",
                value=winner_text,
                normalized_value=winner_text,
                field_type=FieldType.ORGANIZATION,
                provenance=prov,
                extraction_confidence=min(0.95, winner_score),
                is_required=True,
            ), None

        return None, None

    def _extract_customer_info(
        self,
        document: Document,
    ) -> Tuple[Optional[ExtractedField], Optional[ExtractedField]]:
        if not document.pages:
            return None, None

        for page in document.pages:
            matches = find_label_value_matches(
                page,
                INVOICE_FIELD_LABELS["customer_name"],
                self.config.spatial,
            )

            for m in matches:
                val = normalize_text(m.value_text)

                if val and not is_invalid_party_name(val):
                    prov = Provenance(
                        document_id=document.id,
                        page_number=page.page_number,
                        raw_text=m.source_line,
                        bounding_box=m.value_bbox,
                        ocr_confidence=m.mean_ocr_confidence,
                    )
                    return ExtractedField(
                        name="customer_name",
                        value=val,
                        normalized_value=val,
                        field_type=FieldType.ORGANIZATION,
                        provenance=prov,
                        extraction_confidence=0.88,
                        matched_label=m.label_text,
                    ), None

            lines = [
                normalize_text(line)
                for line in (page.raw_text or "").split("\n")
                if normalize_text(line)
            ]

            for idx, line in enumerate(lines):
                line_lower = line.lower().strip().rstrip(":")

                if any(
                    line_lower.startswith(lbl.lower().rstrip(":"))
                    for lbl in INVOICE_FIELD_LABELS["customer_name"]
                ):
                    for next_idx in range(idx + 1, min(len(lines), idx + 4)):
                        next_line = lines[next_idx]

                        if next_line and not is_invalid_party_name(next_line):
                            box, ocr_c = locate_value_in_regions(
                                next_line, page.ocr_text_regions
                            )
                            prov = Provenance(
                                document_id=document.id,
                                page_number=page.page_number,
                                raw_text=next_line,
                                bounding_box=box,
                                ocr_confidence=ocr_c,
                                extraction_method=ExtractionMethod.LAYOUT_ANALYSIS,
                            )
                            return ExtractedField(
                                name="customer_name",
                                value=next_line,
                                normalized_value=next_line,
                                field_type=FieldType.ORGANIZATION,
                                provenance=prov,
                                extraction_confidence=0.85,
                                matched_label=line,
                            ), None

        return None, None

    def _infer_currency(
        self,
        extracted_fields: Dict[str, ExtractedField],
        document: Document,
    ) -> Optional[ExtractedField]:
        for fname in ("total", "subtotal", "tax"):
            field = extracted_fields.get(fname)

            if field and field.metadata.get("currency"):
                curr = field.metadata["currency"]
                prov = Provenance(
                    document_id=document.id,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    bounding_box=field.bounding_box,
                    ocr_confidence=field.source_ocr_confidence,
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
            matches = find_label_value_matches(
                page, labels, self.config.spatial
            )

            if matches:
                match = matches[0]

                if field_type == FieldType.IDENTIFIER:
                    norm_val = normalize_identifier(match.value_text)
                    raw_val = norm_val
                else:
                    norm_val = normalize_text(match.value_text)
                    raw_val = match.value_text

                    # Remove known sample disclaimers from payment terms only.
                    if field_name == "payment_terms":
                        disclaimer_markers = [
                            "This is a fictional sample invoice",
                            "This document is for testing",
                        ]

                        for marker in disclaimer_markers:
                            index = norm_val.lower().find(
                                marker.lower()
                            )

                            if index != -1:
                                norm_val = norm_val[:index].strip()
                                raw_val = norm_val
                                break

                if not norm_val:
                    continue

                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=match.source_line,
                    bounding_box=match.value_bbox,
                    ocr_confidence=match.mean_ocr_confidence,
                )

                return ExtractedField(
                    name=field_name,
                    value=raw_val,
                    normalized_value=norm_val,
                    field_type=field_type,
                    provenance=prov,
                    extraction_confidence=0.85,
                    matched_label=match.label_text,
                )

        return None


    def _extract_tax_id(self, document: Document) -> Optional[ExtractedField]:
        tax_field = self._extract_text_field(
            document,
            "tax_id",
            INVOICE_FIELD_LABELS["tax_id"],
            FieldType.IDENTIFIER,
        )

        if tax_field and tax_field.normalized_value:
            isolated = isolate_tax_identifier(str(tax_field.normalized_value))

            if isolated:
                tax_field.value = isolated
                tax_field.normalized_value = isolated
                return tax_field

        for page in document.pages:
            isolated = isolate_tax_identifier(page.raw_text)

            if isolated:
                box, ocr_c = locate_value_in_regions(
                    isolated, page.ocr_text_regions
                )
                prov = Provenance(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=isolated,
                    bounding_box=box,
                    ocr_confidence=ocr_c,
                    extraction_method=ExtractionMethod.REGEX_PATTERN,
                )
                return ExtractedField(
                    name="tax_id",
                    value=isolated,
                    normalized_value=isolated,
                    field_type=FieldType.IDENTIFIER,
                    provenance=prov,
                    extraction_confidence=0.85,
                )

        return None