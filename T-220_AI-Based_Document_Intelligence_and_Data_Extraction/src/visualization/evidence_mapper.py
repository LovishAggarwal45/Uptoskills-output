"""Evidence mapper compiling raw pipeline artifacts into structured EvidenceRegions and OverlayAnnotations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.core.models import BoundingBox, Document, ExtractedField
from src.core.types import SeverityLevel, ValidationStatus
from src.confidence.models import (
    ConfidenceBand,
    DocumentConfidence,
    FieldConfidence,
    ReviewQueueItem,
    ReviewRoutingResult,
    TableConfidence,
)
from src.extraction.models import ExtractedEntity, ExtractionResult
from src.ocr.models import DocumentOCRResult, OCRWord
from src.tables.models import Table
from src.validation.models import ValidationIssue, ValidationReport
from src.visualization.field_overlay import FieldOverlayBuilder
from src.visualization.models import (
    AnnotationType,
    EvidenceRegion,
    OverlayAnnotation,
    VisualizationManifest,
)
from src.visualization.review_overlay import ReviewOverlayBuilder
from src.visualization.table_overlay import TableOverlayBuilder

logger = get_logger("visualization.mapper")


class EvidenceMapper:
    """Extracts and consolidates evidence provenance across OCR, extraction, tables, validation, and review routing."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id

    def map_all(
        self,
        document: Document,
        fields: Optional[Dict[str, ExtractedField]] = None,
        entities: Optional[List[ExtractedEntity]] = None,
        tables: Optional[List[Table]] = None,
        validation_report: Optional[ValidationReport] = None,
        routing_result: Optional[ReviewRoutingResult] = None,
        ocr_result: Optional[DocumentOCRResult] = None,
    ) -> Tuple[VisualizationManifest, Dict[int, List[OverlayAnnotation]]]:
        """Compile complete evidence manifest and build page-specific overlay annotations.

        Args:
            document: Processed Document aggregate.
            fields: Optional map of extracted fields from Phase 5.
            entities: Optional list of extracted entities.
            tables: Optional list of tables from Phase 6.
            validation_report: Optional validation report from Phase 7.
            routing_result: Optional confidence scoring & review routing result from Phase 8.
            ocr_result: Optional document OCR result from Phase 3.

        Returns:
            Tuple of (VisualizationManifest, Dict[page_number, List[OverlayAnnotation]]).
        """
        spatial_evidence: List[EvidenceRegion] = []
        unlocated_evidence: List[EvidenceRegion] = []

        fields = fields or {}
        entities = entities or []
        tables = tables or []

        # 1. Map OCR Evidence
        ocr_regions = self._map_ocr_evidence(document, ocr_result)
        spatial_evidence.extend(ocr_regions)

        # 2. Map Structured Field Evidence
        field_confs = routing_result.field_confidences if routing_result else {}
        field_spatial, field_unlocated = self._map_field_evidence(fields, field_confs)
        spatial_evidence.extend(field_spatial)
        unlocated_evidence.extend(field_unlocated)

        # 3. Map Generic Entity Evidence
        entity_spatial = self._map_entity_evidence(entities)
        spatial_evidence.extend(entity_spatial)

        # 4. Map Table Evidence
        table_confs = routing_result.table_confidences if routing_result else []
        table_spatial = self._map_table_evidence(tables, table_confs)
        spatial_evidence.extend(table_spatial)

        # 5. Map Validation Issue Evidence
        val_spatial, val_unlocated = self._map_validation_evidence(validation_report)
        spatial_evidence.extend(val_spatial)
        unlocated_evidence.extend(val_unlocated)

        # 6. Map Review Target Evidence
        rev_spatial, rev_unlocated = self._map_review_evidence(routing_result)
        spatial_evidence.extend(rev_spatial)
        unlocated_evidence.extend(rev_unlocated)

        # Assemble Document Manifest
        manifest = VisualizationManifest(
            document_id=self.document_id,
            total_pages=len(document.pages),
            evidence_regions=spatial_evidence,
            unlocated_evidence=unlocated_evidence,
            metadata={
                "spatial_count": len(spatial_evidence),
                "unlocated_count": len(unlocated_evidence),
            },
        )

        # 7. Generate Page-Level Annotations
        annotations_by_page = self._build_page_annotations(
            manifest=manifest,
            pages=document.pages,
        )

        logger.info(
            f"Evidence mapping complete for '{self.document_id}': "
            f"{len(spatial_evidence)} spatial regions, {len(unlocated_evidence)} unlocated issues."
        )

        return manifest, annotations_by_page

    def _map_ocr_evidence(
        self,
        document: Document,
        ocr_result: Optional[DocumentOCRResult] = None,
    ) -> List[EvidenceRegion]:
        """Extract word-level OCR evidence from document pages."""
        regions: List[EvidenceRegion] = []

        for page in document.pages:
            # Check for words in metadata or ocr_result
            words: List[OCRWord] = page.metadata.get("ocr_words", [])
            if not words and ocr_result:
                if hasattr(ocr_result, "get_page"):
                    p_res = ocr_result.get_page(page.page_number)
                    if p_res and hasattr(p_res, "words"):
                        words = p_res.words
                elif hasattr(ocr_result, "pages"):
                    for p in ocr_result.pages:
                        if getattr(p, "page_number", None) == page.page_number:
                            words = getattr(p, "words", [])
                            break

            for idx, word in enumerate(words):
                if word.bounding_box:
                    regions.append(
                        EvidenceRegion(
                            evidence_id=f"ev_ocr_p{page.page_number}_w{idx}",
                            document_id=self.document_id,
                            page_number=page.page_number,
                            region_type=AnnotationType.OCR_WORD,
                            bbox=word.bounding_box,
                            source_text=word.text,
                            ocr_confidence=word.confidence,
                            is_spatial=True,
                            metadata={"word_index": idx, "raw_confidence": word.raw_confidence},
                        )
                    )

        return regions

    def _map_field_evidence(
        self,
        fields: Dict[str, ExtractedField],
        field_confs: Dict[str, FieldConfidence],
    ) -> Tuple[List[EvidenceRegion], List[EvidenceRegion]]:
        """Extract evidence for structured key-value fields."""
        spatial: List[EvidenceRegion] = []
        unlocated: List[EvidenceRegion] = []

        for name, field_obj in fields.items():
            fc = field_confs.get(name)
            band = fc.confidence_band if fc else None
            agg_conf = fc.aggregated_confidence if fc else field_obj.extraction_confidence

            if field_obj.bounding_box:
                spatial.append(
                    EvidenceRegion(
                        evidence_id=f"ev_fld_{name}",
                        document_id=self.document_id,
                        page_number=field_obj.page_number,
                        region_type=AnnotationType.EXTRACTED_FIELD,
                        bbox=field_obj.bounding_box,
                        source_text=field_obj.source_text or str(field_obj.value or ""),
                        normalized_value=field_obj.normalized_value,
                        field_name=name,
                        ocr_confidence=field_obj.source_ocr_confidence,
                        extraction_confidence=field_obj.extraction_confidence,
                        aggregated_confidence=agg_conf,
                        confidence_band=band,
                        validation_status=field_obj.validation_status,
                        review_required=field_obj.validation_status == ValidationStatus.INVALID or (fc.review_required if fc else False),
                        review_reasons=list(field_obj.validation_messages),
                        source_method=field_obj.extraction_method.value if field_obj.extraction_method else None,
                        provenance=field_obj.provenance,
                        is_spatial=True,
                    )
                )
            else:
                unlocated.append(
                    EvidenceRegion(
                        evidence_id=f"ev_fld_{name}_unlocated",
                        document_id=self.document_id,
                        page_number=field_obj.page_number,
                        region_type=AnnotationType.EXTRACTED_FIELD,
                        bbox=None,
                        source_text=str(field_obj.value or ""),
                        normalized_value=field_obj.normalized_value,
                        field_name=name,
                        extraction_confidence=field_obj.extraction_confidence,
                        aggregated_confidence=agg_conf,
                        confidence_band=band,
                        validation_status=field_obj.validation_status,
                        review_required=field_obj.validation_status == ValidationStatus.INVALID,
                        review_reasons=list(field_obj.validation_messages),
                        is_spatial=False,
                        metadata={"warning": "Field extracted without spatial bounding box coordinates"},
                    )
                )

        return spatial, unlocated

    def _map_entity_evidence(
        self,
        entities: List[ExtractedEntity],
    ) -> List[EvidenceRegion]:
        """Extract evidence for generic named entities."""
        spatial: List[EvidenceRegion] = []
        for idx, ent in enumerate(entities):
            if ent.bounding_box:
                etype = ent.entity_type.value if hasattr(ent.entity_type, "value") else str(ent.entity_type)
                src_txt = getattr(ent, "source_text", None) or getattr(ent, "raw_text", None) or getattr(ent, "value", "")
                c_span = getattr(ent, "character_span", None) or getattr(ent, "char_span", None)
                spatial.append(
                    EvidenceRegion(
                        evidence_id=f"ev_ent_{etype}_{idx}",
                        document_id=self.document_id,
                        page_number=ent.page_number,
                        region_type=AnnotationType.EXTRACTED_ENTITY,
                        bbox=ent.bounding_box,
                        source_text=src_txt,
                        normalized_value=ent.normalized_value,
                        field_name=etype,
                        extraction_confidence=ent.confidence,
                        aggregated_confidence=ent.confidence,
                        is_spatial=True,
                        metadata={"entity_type": etype, "char_span": c_span},
                    )
                )
        return spatial

    def _map_table_evidence(
        self,
        tables: List[Table],
        table_confs: List[TableConfidence],
    ) -> List[EvidenceRegion]:
        """Extract evidence for tables, rows, cells, and line items."""
        spatial: List[EvidenceRegion] = []

        for tbl_idx, tbl in enumerate(tables):
            # 1. Outer table bounding box
            if tbl.bounding_box:
                spatial.append(
                    EvidenceRegion(
                        evidence_id=f"ev_tbl_{tbl.table_id}_bounds",
                        document_id=self.document_id,
                        page_number=tbl.page_number,
                        region_type=AnnotationType.TABLE_BOUNDS,
                        bbox=tbl.bounding_box,
                        table_id=tbl.table_id,
                        extraction_confidence=tbl.confidence,
                        aggregated_confidence=tbl.confidence,
                        is_spatial=True,
                        metadata={"header_count": len(tbl.headers), "row_count": len(tbl.rows)},
                    )
                )

            # 2. Table Cells
            for row in tbl.rows:
                for cell in row.cells:
                    if cell.bounding_box:
                        col_name = tbl.headers[cell.col_index] if cell.col_index < len(tbl.headers) else f"col_{cell.col_index}"
                        spatial.append(
                            EvidenceRegion(
                                evidence_id=f"ev_tbl_{tbl.table_id}_r{cell.row_index}_c{cell.col_index}",
                                document_id=self.document_id,
                                page_number=cell.page_number,
                                region_type=AnnotationType.TABLE_CELL,
                                bbox=cell.bounding_box,
                                source_text=cell.text,
                                normalized_value=cell.normalized_value,
                                table_id=tbl.table_id,
                                row_index=cell.row_index,
                                column_name=col_name,
                                ocr_confidence=cell.confidence,
                                extraction_confidence=cell.confidence,
                                is_spatial=True,
                            )
                        )

            # 3. Line Items & Math Validation
            for li in tbl.line_items:
                reasons = []
                val_status = ValidationStatus.VALID if li.is_valid_arithmetic else ValidationStatus.INVALID
                if not li.is_valid_arithmetic:
                    reasons.append(f"Line item math mismatch: Qty {li.quantity} * Price {li.unit_price} != Amount {li.amount}")

                if tbl.bounding_box:
                    spatial.append(
                        EvidenceRegion(
                            evidence_id=f"ev_tbl_{tbl.table_id}_item_{li.row_index}",
                            document_id=self.document_id,
                            page_number=li.page_number,
                            region_type=AnnotationType.TABLE_LINE_ITEM,
                            bbox=tbl.bounding_box,
                            source_text=li.description,
                            normalized_value=li.amount,
                            table_id=tbl.table_id,
                            row_index=li.row_index,
                            extraction_confidence=li.confidence,
                            validation_status=val_status,
                            review_required=not li.is_valid_arithmetic,
                            review_reasons=reasons,
                            is_spatial=True,
                        )
                    )

        return spatial

    def _map_validation_evidence(
        self,
        report: Optional[ValidationReport],
    ) -> Tuple[List[EvidenceRegion], List[EvidenceRegion]]:
        """Extract evidence for validation findings."""
        spatial: List[EvidenceRegion] = []
        unlocated: List[EvidenceRegion] = []

        if not report:
            return spatial, unlocated

        for idx, issue in enumerate(report.issues):
            is_crit = issue.severity == SeverityLevel.CRITICAL
            reg_type = (
                AnnotationType.VALIDATION_CRITICAL if is_crit
                else (AnnotationType.VALIDATION_ERROR if issue.status == ValidationStatus.INVALID
                      else AnnotationType.VALIDATION_WARNING)
            )

            if issue.bounding_box:
                spatial.append(
                    EvidenceRegion(
                        evidence_id=f"ev_val_{issue.rule_id}_{idx}",
                        document_id=self.document_id,
                        page_number=issue.page_number or 1,
                        region_type=reg_type,
                        bbox=issue.bounding_box,
                        source_text=getattr(issue, "raw_text", None) or getattr(issue, "source_text", None) or getattr(issue, "message", ""),
                        field_name=issue.affected_fields[0] if issue.affected_fields else None,
                        validation_status=issue.status,
                        review_required=issue.status == ValidationStatus.INVALID,
                        review_reasons=[issue.message],
                        is_spatial=True,
                        metadata={"rule_name": issue.rule_name, "severity": issue.severity.value},
                    )
                )
            else:
                unlocated.append(
                    EvidenceRegion(
                        evidence_id=f"ev_val_{issue.rule_id}_{idx}_unlocated",
                        document_id=self.document_id,
                        page_number=issue.page_number or 1,
                        region_type=reg_type,
                        bbox=None,
                        field_name=issue.affected_fields[0] if issue.affected_fields else None,
                        validation_status=issue.status,
                        review_required=issue.status == ValidationStatus.INVALID,
                        review_reasons=[issue.message],
                        is_spatial=False,
                        metadata={"rule_name": issue.rule_name, "severity": issue.severity.value},
                    )
                )

        return spatial, unlocated

    def _map_review_evidence(
        self,
        routing_result: Optional[ReviewRoutingResult],
    ) -> Tuple[List[EvidenceRegion], List[EvidenceRegion]]:
        """Extract evidence for human review targets."""
        spatial: List[EvidenceRegion] = []
        unlocated: List[EvidenceRegion] = []

        if not routing_result or not routing_result.queue_item:
            return spatial, unlocated

        queue_item = routing_result.queue_item

        for idx, target in enumerate(queue_item.targets):
            is_crit = target.severity == SeverityLevel.CRITICAL
            reg_type = AnnotationType.REVIEW_CRITICAL if is_crit else AnnotationType.REVIEW_TARGET

            if target.bounding_box:
                spatial.append(
                    EvidenceRegion(
                        evidence_id=f"ev_rev_target_{idx}",
                        document_id=self.document_id,
                        page_number=target.page_number,
                        region_type=reg_type,
                        bbox=target.bounding_box,
                        field_name=target.field_name,
                        table_id=target.table_id,
                        row_index=target.row_index,
                        column_name=target.column_name,
                        validation_status=ValidationStatus.INVALID if is_crit else ValidationStatus.WARNING,
                        review_required=True,
                        review_reasons=[target.reason] if target.reason else [],
                        is_spatial=True,
                        metadata={"target_type": target.target_type.value},
                    )
                )
            else:
                unlocated.append(
                    EvidenceRegion(
                        evidence_id=f"ev_rev_target_{idx}_unlocated",
                        document_id=self.document_id,
                        page_number=target.page_number,
                        region_type=reg_type,
                        bbox=None,
                        field_name=target.field_name,
                        table_id=target.table_id,
                        review_required=True,
                        review_reasons=[target.reason] if target.reason else [],
                        is_spatial=False,
                        metadata={"target_type": target.target_type.value},
                    )
                )

        return spatial, unlocated

    def _build_page_annotations(
        self,
        manifest: VisualizationManifest,
        pages: List[Any],
    ) -> Dict[int, List[OverlayAnnotation]]:
        """Compile raw EvidenceRegions into sorted, non-overlapping page-level OverlayAnnotations."""
        annotations_by_page: Dict[int, List[OverlayAnnotation]] = {}

        # Group spatial evidence by page
        evidence_by_page: Dict[int, List[EvidenceRegion]] = {}
        for p in pages:
            evidence_by_page[p.page_number] = []
            annotations_by_page[p.page_number] = []

        for er in manifest.evidence_regions:
            if er.page_number in evidence_by_page:
                evidence_by_page[er.page_number].append(er)

        for p_num, p_regions in evidence_by_page.items():
            # Build using specialized builders
            fld_anns = FieldOverlayBuilder.build_all(p_regions)
            tbl_anns = TableOverlayBuilder.build_all(p_regions)
            rev_anns = ReviewOverlayBuilder.build_all(p_regions)

            # Consolidate and prioritize: Review > Tables > Fields
            all_page_anns = []
            all_page_anns.extend(fld_anns)
            all_page_anns.extend(tbl_anns)
            all_page_anns.extend(rev_anns)

            annotations_by_page[p_num] = all_page_anns

        return annotations_by_page


__all__ = ["EvidenceMapper"]
