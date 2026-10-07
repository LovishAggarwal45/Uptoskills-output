
"""Unified document intelligence extraction dispatcher."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.classification.models import ClassificationResult
from src.core.config import ExtractionConfig
from src.core.logging import get_logger
from src.core.models import Document, ExtractedField
from src.core.types import DocumentType
from src.extraction.base import BaseDocumentExtractor
from src.extraction.entity_extractors import GenericEntityExtractor
from src.extraction.exceptions import InvalidExtractionInput
from src.extraction.form_extractor import FormExtractor
from src.extraction.general_extractor import GeneralDocumentExtractor
from src.extraction.invoice_extractor import InvoiceExtractor
from src.extraction.models import ExtractedEntity, ExtractionResult
from src.extraction.receipt_extractor import ReceiptExtractor
from src.extraction.resume_extractor import ResumeExtractor

logger = get_logger("extraction")


class DocumentExtractor(BaseDocumentExtractor):
    """Coordinate document-specific and generic entity extraction."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        super().__init__(config=config)

        self.generic_entity_extractor = GenericEntityExtractor(
            config=self.config
        )
        self.invoice_extractor = InvoiceExtractor(config=self.config)
        self.receipt_extractor = ReceiptExtractor(config=self.config)
        self.form_extractor = FormExtractor(config=self.config)
        self.resume_extractor = ResumeExtractor(config=self.config)
        self.general_extractor = GeneralDocumentExtractor(
            config=self.config
        )

    @staticmethod
    def _get_document_text(document: Document) -> str:
        """Combine raw OCR text and OCR regions from every page."""
        page_texts = []

        for page in document.pages:
            raw_text = page.raw_text or ""

            region_text = " ".join(
                region.text
                for region in page.ocr_text_regions
                if region.text
            )

            page_texts.append(f"{raw_text}\n{region_text}")

        return "\n".join(page_texts).lower()

    @classmethod
    def _is_railway_ticket(cls, document: Document) -> bool:
        """Identify railway reservation slips using document text."""
        text = cls._get_document_text(document)

        has_ticket_heading = (
            "electronic reservation slip" in text
            or "electronic reservation slip (ers)" in text
        )

        has_railway_issuer = (
            "indian railways" in text
            or "irctc" in text
        )

        return has_ticket_heading and has_railway_issuer

    def extract(
        self,
        document: Document,
        classification_result: Optional[ClassificationResult] = None,
    ) -> ExtractionResult:
        """Extract structured fields and generic entities.

        Args:
            document: Ingested document with pages and OCR results.
            classification_result: Optional classification output.

        Returns:
            ExtractionResult containing fields, entities, warnings,
            unresolved fields, and extraction statistics.

        Raises:
            InvalidExtractionInput: If the document is invalid.
        """
        if document is None or not isinstance(document, Document):
            raise InvalidExtractionInput(
                "Invalid input: document must be an instance of Document."
            )

        start_time = time.perf_counter()

        # Determine the effective document category.
        doc_type = DocumentType.UNKNOWN

        if (
            classification_result
            and classification_result.document_type
        ):
            doc_type = classification_result.document_type
        elif (
            document.classified_type
            and document.classified_type != DocumentType.UNKNOWN
        ):
            doc_type = document.classified_type

        # Check the actual document content, not only its category.
        is_railway_ticket = self._is_railway_ticket(document)

        logger.info(
            "Extracting structured data for document '%s' "
            "(Category: %s, Railway ticket: %s)",
            document.id,
            doc_type.value.upper(),
            is_railway_ticket,
        )

        extracted_fields: Dict[str, ExtractedField] = {}
        unresolved_fields: List[str] = []
        warnings: List[str] = []

        # 1. Document-specific field extraction.
        # Railway slips must not be processed as ordinary invoices,
        # receipts, forms, resumes, or generic business documents.
        if is_railway_ticket:
            warnings.append(
                "Railway reservation slip detected. "
                "Generic document fields were skipped because "
                "a dedicated railway-ticket extractor is not "
                "yet implemented."
            )

        elif doc_type == DocumentType.INVOICE:
            extracted_fields = self.invoice_extractor.extract(document)

            required = self.config.invoice_required_fields
            for req_field in required:
                if req_field not in extracted_fields:
                    unresolved_fields.append(req_field)

        elif doc_type == DocumentType.RECEIPT:
            extracted_fields = self.receipt_extractor.extract(document)

            required = self.config.receipt_required_fields
            for req_field in required:
                if req_field not in extracted_fields:
                    unresolved_fields.append(req_field)

        elif doc_type == DocumentType.FORM:
            extracted_fields = self.form_extractor.extract(document)

            required = self.config.form_required_fields
            for req_field in required:
                if req_field not in extracted_fields:
                    unresolved_fields.append(req_field)

        elif doc_type == DocumentType.RESUME:
            extracted_fields = self.resume_extractor.extract(document)

        elif doc_type == DocumentType.GENERAL_DOCUMENT:
            extracted_fields = self.general_extractor.extract(document)

        elif doc_type == DocumentType.UNKNOWN:
            # Do not force a domain-specific schema onto unknown documents.
            pass

        # 2. Universal generic entity extraction.
        # This extracts candidate entities, not verified business fields.
        entities: List[ExtractedEntity] = []

        if not is_railway_ticket:
            entities = self.generic_entity_extractor.extract_entities(
                document
            )

        # 3. Detect ambiguous extracted fields.
        ambiguous_fields = [
            field.name
            for field in extracted_fields.values()
            if field.is_ambiguous
        ]

        if ambiguous_fields:
            warnings.append(
                "Ambiguous candidate values detected for fields: "
                + ", ".join(ambiguous_fields)
            )

        elapsed = time.perf_counter() - start_time

        stats = {
            "total_pages": len(document.pages),
            "total_fields": len(extracted_fields),
            "field_count": len(extracted_fields),
            "total_entities": len(entities),
            "entity_count": len(entities),
            "unresolved_count": len(unresolved_fields),
            "ambiguous_count": len(ambiguous_fields),
            "execution_time_seconds": round(elapsed, 4),
        }

        logger.info(
            "Extraction completed for '%s': %d fields, %d entities "
            "(%.1f ms)",
            document.id,
            len(extracted_fields),
            len(entities),
            elapsed * 1000,
        )

        return ExtractionResult(
            document_id=document.id,
            document_type=doc_type,
            fields=extracted_fields,
            entities=entities,
            unresolved_fields=unresolved_fields,
            warnings=warnings,
            extraction_statistics=stats,
            extractor_name="rule_based_extractor",
            extractor_version="1.0.0",
            execution_time_seconds=elapsed,
            metadata={
                "classification_method": (
                    classification_result.method
                    if classification_result
                    else "none"
                ),
                "confidence_threshold": self.config.confidence_threshold,
            },
        )

    def extract_document(
        self,
        document: Document,
        classification_result: Optional[ClassificationResult] = None,
    ) -> ExtractionResult:
        """Alias for extract()."""
        return self.extract(
            document,
            classification_result=classification_result,
        )


RuleBasedDocumentExtractor = DocumentExtractor

__all__ = [
    "DocumentExtractor",
    "RuleBasedDocumentExtractor",
]
