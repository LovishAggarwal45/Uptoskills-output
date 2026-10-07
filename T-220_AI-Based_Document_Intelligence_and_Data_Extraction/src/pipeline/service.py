
"""Unified document processing pipeline service coordinating all intelligence subsystems and persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.core.config import DocuMindConfig, load_config
from src.core.logging import get_logger
from src.core.models import ProcessingResult, ProcessingSummary
from src.classification.classifier import RuleBasedDocumentClassifier
from src.confidence.review_router import ReviewRouter
from src.export.exporter import DocumentExporter
from src.extraction.extractor import DocumentExtractor
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.processor import DocumentOCRProcessor
from src.persistence.models import DocumentProcessingStatus, DocumentRecord
from src.persistence.repository import AuditRepository, DocumentRepository
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor
from src.tables.extractor import TableExtractor
from src.validation.validator import DocumentValidationEngine
from src.visualization.base import DocumentVisualizer

logger = get_logger("pipeline.service")


class DocumentPipelineService:
    """End-to-end execution service that executes the full DocuMind AI intelligence pipeline."""

    def __init__(
        self,
        config: Optional[DocuMindConfig] = None,
        doc_repo: Optional[DocumentRepository] = None,
        audit_repo: Optional[AuditRepository] = None,
    ) -> None:
        self.config = config or load_config()
        self.doc_repo = doc_repo or DocumentRepository()
        self.audit_repo = audit_repo or AuditRepository()
        self.exporter = DocumentExporter(config=self.config.storage)

    def process_document(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        actor: str = "system",
    ) -> ProcessingResult:
        """Execute full 10-stage processing pipeline on a target document.

        Args:
            file_path: Path to the input file (PDF or supported image).
            document_id: Optional predefined document ID.
            actor: Identity of actor triggering processing.

        Returns:
            Fully populated ProcessingResult.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input document not found at {path}")

        start_time = time.perf_counter()
        logger.info(f"Starting unified pipeline processing for: {path.name}")

        # Ensure working output directories exist
        processed_base = Path("data/processed")
        vis_base = Path("outputs/visualizations")
        processed_base.mkdir(parents=True, exist_ok=True)
        vis_base.mkdir(parents=True, exist_ok=True)

        # 1. Ingestion
        loader = DocumentLoader(
            config=self.config.ingestion,
            processed_dir=processed_base,
        )
        doc = loader.ingest(path, output_dir=processed_base)

        if document_id:
            doc.id = document_id
            doc.metadata.document_id = document_id

        doc_id = doc.id
        existing_doc = self.doc_repo.get_by_id(doc_id)

        if not existing_doc:
            file_stat = path.stat() if path.exists() else None
            self.doc_repo.create(
                DocumentRecord(
                    document_id=doc_id,
                    filename=path.name,
                    file_type=path.suffix.lstrip(".").lower(),
                    file_size_bytes=file_stat.st_size if file_stat else 0,
                    checksum_sha256="",
                    page_count=len(doc.pages),
                    status=DocumentProcessingStatus.PROCESSING,
                    storage_path=str(path),
                )
            )
        else:
            self.doc_repo.update_status(
                doc_id,
                DocumentProcessingStatus.PROCESSING,
            )

        self.audit_repo.log_event(
            doc_id,
            "PIPELINE_STARTED",
            f"Started processing document '{path.name}' ({len(doc.pages)} pages)",
            actor=actor,
        )

        try:
            # 2. Image Preprocessing
            doc_processed_dir = processed_base / doc_id
            doc_processed_dir.mkdir(parents=True, exist_ok=True)

            preprocessor = OpenCVImagePreprocessor(
                config=self.config.preprocessing,
            )
            for page in doc.pages:
                prep_res = preprocessor.preprocess_page(
                    page,
                    output_dir=doc_processed_dir,
                )
                page.image_path = prep_res.processed_image_path
                page.metadata["deskew_angle"] = prep_res.deskew_angle
                page.metadata["quality_metrics"] = prep_res.metrics

            self.audit_repo.log_event(
                doc_id,
                "PREPROCESSING_COMPLETED",
                f"Preprocessed {len(doc.pages)} pages (deskew, contrast enhancement)",
                actor=actor,
            )

            # 3. OCR Subsystem
            ocr_avail = check_ocr_availability(self.config.ocr)
            ocr_result = None

            if ocr_avail.is_ready:
                ocr_proc = DocumentOCRProcessor(config=self.config.ocr)
                ocr_result = ocr_proc.process_document(
                    doc,
                    processed_dir=processed_base,
                )
                self.audit_repo.log_event(
                    doc_id,
                    "OCR_COMPLETED",
                    (
                        f"OCR recognized {ocr_result.total_words} words "
                        f"across {len(doc.pages)} pages "
                        f"(Engine: {ocr_result.engine_name})"
                    ),
                    actor=actor,
                )
            else:
                logger.warning(
                    "Tesseract OCR is not available. Proceeding with existing page text."
                )
                self.audit_repo.log_event(
                    doc_id,
                    "OCR_SKIPPED",
                    "OCR engine binary unavailable in environment; using fallback text buffer",
                    actor=actor,
                )

            # 4. Document Classification
            classifier = RuleBasedDocumentClassifier(
                config=self.config.classification,
            )
            clf_result = classifier.classify(doc)
            doc.classified_type = clf_result.document_type
            doc.classification_confidence = clf_result.confidence

            self.audit_repo.log_event(
                doc_id,
                "CLASSIFICATION_COMPLETED",
                (
                    f"Classified as '{doc.classified_type.value.upper()}' "
                    f"with {doc.classification_confidence:.2%} confidence"
                ),
                actor=actor,
            )

            # 5. Field & Entity Extraction
            extractor = DocumentExtractor(config=self.config.extraction)
            ext_result = extractor.extract(doc, clf_result)
            fields = ext_result.fields
            entities = ext_result.entities

            self.audit_repo.log_event(
                doc_id,
                "EXTRACTION_COMPLETED",
                f"Extracted {len(fields)} structured fields and {len(entities)} named entities",
                actor=actor,
            )

            # 6. Tabular Line-Item Extraction
            doc_totals: Dict[str, float] = {}

            if (
                "total" in fields
                and isinstance(fields["total"].normalized_value, (int, float))
            ):
                doc_totals["total"] = float(fields["total"].normalized_value)

            if (
                "subtotal" in fields
                and isinstance(fields["subtotal"].normalized_value, (int, float))
            ):
                doc_totals["subtotal"] = float(fields["subtotal"].normalized_value)

            if (
                "tax" in fields
                and isinstance(fields["tax"].normalized_value, (int, float))
            ):
                doc_totals["tax"] = float(fields["tax"].normalized_value)

            table_extractor = TableExtractor(config=self.config.tables)
            tbl_result = table_extractor.extract(
                doc,
                document_totals=doc_totals if doc_totals else None,
            )
            tables = tbl_result.tables

            self.audit_repo.log_event(
                doc_id,
                "TABLE_EXTRACTION_COMPLETED",
                (
                    f"Extracted {len(tables)} tables with "
                    f"{sum(len(t.line_items) for t in tables)} line items"
                ),
                actor=actor,
            )

            # 7. Document Validation Engine
            val_engine = DocumentValidationEngine(config=self.config.validation)
            val_report = val_engine.validate_document(
                document_id=doc_id,
                document_type=doc.classified_type,
                fields=fields,
                tables=tables,
                context={
                    "rich_tables": tables,
                    "line_items": tbl_result.line_items,
                },
            )

            # Report successful checks and findings separately.
            self.audit_repo.log_event(
                doc_id,
                "VALIDATION_COMPLETED",
                (
                    f"Evaluated {val_report.rules_evaluated} rules: "
                    f"{val_report.rules_passed} passed, "
                    f"{val_report.rules_failed} failed, "
                    f"{val_report.rules_warning} warnings"
                ),
                actor=actor,
            )

            # 8. Confidence Scoring & Review Routing
            router = ReviewRouter(config=self.config.confidence)
            routing_result = router.route_document(
                document=doc,
                fields=fields,
                tables=tables,
                validation_report=val_report,
            )

            self.audit_repo.log_event(
                doc_id,
                "ROUTING_COMPLETED",
                (
                    f"Document score: "
                    f"{routing_result.document_confidence.overall_confidence:.2%} "
                    f"[{routing_result.document_confidence.confidence_band.value.upper()}], "
                    f"Status: {routing_result.queue_item.status.value.upper()}"
                ),
                actor=actor,
            )

            # 9. Visual Explainability & Evidence Overlays
            visualizer = DocumentVisualizer(
                config=self.config.visualization,
            )
            vis_result = visualizer.visualize_document(
                document=doc,
                fields=fields,
                entities=entities,
                tables=tables,
                validation_report=val_report,
                routing_result=routing_result,
                ocr_result=ocr_result,
                output_dir=vis_base,
            )

            # 10. Synthesize Result & Export Artifacts
            now = time.perf_counter()
            total_duration = now - start_time

            # Build ProcessingSummary
            summary = ProcessingSummary(
                document_id=doc_id,
                document_type=doc.classified_type,
                total_pages=len(doc.pages),
                total_fields_extracted=len(fields),
                total_tables_extracted=len(tables),
                validation_passed_count=val_report.rules_passed,
                validation_warning_count=sum(
                    1
                    for i in val_report.issues
                    if getattr(i.status, "value", str(i.status)).lower() == "warning"
                ),
                validation_error_count=sum(
                    1
                    for i in val_report.issues
                    if getattr(i.status, "value", str(i.status)).lower()
                    in ("invalid", "error", "critical")
                ),
                overall_confidence_score=(
                    routing_result.document_confidence.overall_confidence
                ),
                review_required=routing_result.review_required,
                processing_time_seconds=round(total_duration, 4),
            )

            # Build ProcessingResult
            result = ProcessingResult(
                document=doc,
                fields=fields,
                tables=tables,
                validation_results=val_report.issues,
                review_flags=[],
                summary=summary,
                confidence=routing_result.document_confidence,
                review=routing_result,
                visualization=vis_result,
                execution_metadata={
                    "total_duration_seconds": round(total_duration, 4),
                    "ocr_engine": ocr_result.engine_name if ocr_result else "none",
                },
            )

            # Export JSON, CSVs, and Report
            json_path = self.exporter.export_json(result)
            self.exporter.export_csv(result)
            self.exporter.export_validation_report(result)

            # Update Database
            final_status = (
                DocumentProcessingStatus.NEEDS_REVIEW
                if routing_result.review_required
                else DocumentProcessingStatus.COMPLETED
            )
            prio_val = (
                routing_result.queue_item.priority.value
                if routing_result.queue_item
                else "low"
            )
            band_val = (
                routing_result.document_confidence.confidence_band.value
            )
            val_stat_val = val_report.overall_status.value

            self.doc_repo.update_processing_results(
                document_id=doc_id,
                document_type=doc.classified_type.value,
                status=final_status,
                confidence_score=(
                    routing_result.document_confidence.overall_confidence
                ),
                confidence_band=band_val,
                validation_status=val_stat_val,
                review_priority=prio_val,
                review_status=(
                    "needs_review"
                    if routing_result.review_required
                    else "approved"
                ),
                target_count=(
                    routing_result.queue_item.target_count
                    if routing_result.queue_item
                    else 0
                ),
                result_json_path=str(json_path),
                metadata_json=json.dumps(result.to_dict(), default=str),
            )

            self.audit_repo.log_event(
                doc_id,
                "PIPELINE_COMPLETED",
                (
                    f"Successfully finished processing in {total_duration:.3f}s. "
                    f"Final status: {final_status.value.upper()}"
                ),
                actor=actor,
            )

            logger.info(
                f"Pipeline finished successfully for '{doc_id}' "
                f"in {total_duration:.3f}s"
            )
            return result

        except Exception as e:
            err_str = f"Pipeline execution failed: {str(e)}"
            logger.error(err_str, exc_info=True)
            self.doc_repo.update_status(
                doc_id,
                DocumentProcessingStatus.FAILED,
                error_message=str(e),
            )
            self.audit_repo.log_event(
                doc_id,
                "PIPELINE_FAILED",
                err_str,
                actor=actor,
            )
            raise


__all__ = ["DocumentPipelineService"]
