"""Production-quality, explainable rule-based document classifier."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.classification.base import BaseDocumentClassifier
from src.classification.exceptions import (
    ClassificationConfigurationError,
    InvalidClassificationInput,
)
from src.classification.models import (
    ClassificationCandidate,
    ClassificationEvidence,
    ClassificationResult,
    SignalType,
)
from src.classification.rules import (
    RuleDefinition,
    get_default_classification_rules,
    locate_text_provenance,
)
from src.core.config import ClassificationConfig
from src.core.logging import get_logger
from src.core.models import Document
from src.core.types import DocumentType

logger = get_logger("classification")


class RuleBasedDocumentClassifier(BaseDocumentClassifier):
    """Deterministic, rule-based document classifier with provenance and explainability.

    Evaluates structured lexical, pattern, phrase, and negative signals against OCR text
    and page regions to classify documents into standard categories (INVOICE, RECEIPT,
    FORM, GENERAL_DOCUMENT, UNKNOWN).
    """

    def __init__(
        self,
        config: Optional[ClassificationConfig] = None,
        rules: Optional[List[RuleDefinition]] = None,
    ) -> None:
        super().__init__(config=config)
        self.rules: List[RuleDefinition] = []

        if rules is not None:
            self.rules = list(rules)
        elif self.config.keyword_weights_path:
            self.rules = self._load_rules_from_file(self.config.keyword_weights_path)
        else:
            self.rules = get_default_classification_rules()

        # Merge any custom rules specified in configuration
        if self.config.custom_rules:
            self._merge_custom_rules(self.config.custom_rules)

    def _load_rules_from_file(self, file_path: Union[str, Path]) -> List[RuleDefinition]:
        """Load external classification rules and keyword weights from a YAML configuration file."""
        path = Path(file_path)
        if not path.exists():
            raise ClassificationConfigurationError(f"Classification rules file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return self._parse_rules_dict(data)
        except Exception as e:
            raise ClassificationConfigurationError(f"Failed to parse classification rules from {path}: {e}") from e

    def _merge_custom_rules(self, custom_rules: Dict[str, Any]) -> None:
        """Merge custom rules defined in configuration dictionary."""
        additional_rules = self._parse_rules_dict(custom_rules)
        self.rules.extend(additional_rules)

    def _parse_rules_dict(self, data: Dict[str, Any]) -> List[RuleDefinition]:
        """Parse structured dictionary into RuleDefinition list."""
        parsed_rules: List[RuleDefinition] = []
        for doc_type_str, type_rules in data.items():
            try:
                doc_type = DocumentType(doc_type_str.lower())
            except ValueError:
                continue

            if isinstance(type_rules, list):
                for item in type_rules:
                    if isinstance(item, dict):
                        pattern = item.get("pattern") or item.get("keyword") or ""
                        weight = float(item.get("weight", 1.0))
                        is_regex = bool(item.get("is_regex", False))
                        is_negative = bool(item.get("is_negative", weight < 0))
                        sig_type_str = item.get("signal_type", SignalType.KEYWORD.value)
                        name = item.get("name", f"{doc_type.value}:{pattern}")
                        parsed_rules.append(
                            RuleDefinition(
                                name=name,
                                target_type=doc_type,
                                signal_type=SignalType(sig_type_str),
                                pattern=pattern,
                                weight=weight,
                                is_regex=is_regex,
                                is_negative=is_negative,
                                description=item.get("description", ""),
                            )
                        )
                    elif isinstance(item, str):
                        parsed_rules.append(
                            RuleDefinition(
                                name=f"{doc_type.value}:{item}",
                                target_type=doc_type,
                                signal_type=SignalType.KEYWORD,
                                pattern=item,
                                weight=2.0,
                            )
                        )
        return parsed_rules

    def classify(self, document: Document) -> ClassificationResult:
        """Classify document category using rule-based scoring and evidence aggregation.

        Args:
            document: Ingested document aggregate populated with pages and OCR results.

        Returns:
            ClassificationResult containing the classified type, confidence, and explainable evidence.

        Raises:
            InvalidClassificationInput: If the document is None or not a Document instance.
        """
        if document is None or not isinstance(document, Document):
            raise InvalidClassificationInput("Invalid classification input: document must be an instance of Document.")
        
        # Detect railway reservation slips, which may contain invoice and GST details.
        document_text = "\n".join(
            (
                page.raw_text
                or " ".join(
                    region.text for region in page.ocr_text_regions
                )
            )
            for page in document.pages
        ).lower()

        railway_markers = (
            "electronic reservation slip",
            "indian railways",
            "irctc",
            "pnr",
        )

        is_railway_ticket = (
            "electronic reservation slip" in document_text
            and sum(
                marker in document_text
                for marker in railway_markers
            ) >= 3
        )

        start_time = time.perf_counter()
        logger.debug(f"Classifying document '{document.id}' ({len(document.pages)} pages)")

        all_evidence: List[ClassificationEvidence] = []
        supported_types = [
            DocumentType.INVOICE,
            DocumentType.RECEIPT,
            DocumentType.FORM,
            DocumentType.RESUME,
            DocumentType.GENERAL_DOCUMENT,
        ]

        # Process each page
        for page in document.pages:
            page_text = page.raw_text or ""
            if not page_text and page.ocr_text_regions:
                page_text = " ".join(r.text for r in page.ocr_text_regions)

            if not page_text.strip():
                continue

            # Track seen signals per page to avoid excessive duplicate amplification
            seen_page_signals: set = set()

            for rule in self.rules:
                matches = rule.find_matches(page_text)
                for matched_text, (start_span, end_span) in matches:
                    sig_key = (rule.name, matched_text.lower().strip())
                    if sig_key in seen_page_signals:
                        continue
                    seen_page_signals.add(sig_key)

                    # Check for header positioning (top 30% of page 1 text)
                    is_header = (start_span / max(1, len(page_text))) <= 0.30
                    weight = rule.weight
                    if is_header and page.page_number == 1 and weight > 0:
                        weight *= self.config.header_weight_multiplier

                    # Spatial and OCR confidence provenance
                    bbox, ocr_conf = locate_text_provenance(matched_text, page)

                    evidence = ClassificationEvidence(
                        signal_name=rule.name,
                        signal_type=rule.signal_type,
                        matched_text=matched_text,
                        weight=weight,
                        target_type=rule.target_type,
                        page_number=page.page_number,
                        bounding_box=bbox,
                        ocr_confidence=ocr_conf,
                        character_span=(start_span, end_span),
                        details={
                            "is_header": is_header,
                            "rule_description": rule.description,
                            "is_negative": rule.is_negative,
                        },
                    )
                    all_evidence.append(evidence)
        
        if is_railway_ticket:
            logger.info(
                "Railway reservation slip detected; "
                "suppressing invoice-specific evidence."
            )
            all_evidence = [
                evidence
                for evidence in all_evidence
                if not (
                    evidence.target_type == DocumentType.INVOICE
                    and evidence.signal_name in {
                        "invoice:invoice number",
                        "invoice:invoice no",
                        "invoice:invoice #",
                        "invoice:invoice",
                    }
                )
            ]

        # Build Candidate Summaries
        candidates: Dict[str, ClassificationCandidate] = {}
        for target_type in supported_types:
            type_evidence = [e for e in all_evidence if e.target_type == target_type]
            raw_score = sum(e.weight for e in type_evidence)
            raw_score = max(0.0, raw_score)  # Floor at 0
            pos_count = sum(1 for e in type_evidence if e.weight > 0)

            candidates[target_type.value] = ClassificationCandidate(
                document_type=target_type,
                raw_score=raw_score,
                normalized_score=0.0,
                evidence_count=pos_count,
                evidence=type_evidence,
            )

        # Evaluate Top Candidate and Ambiguity
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c.raw_score,
            reverse=True,
        )

        top_cand = sorted_candidates[0] if sorted_candidates else None
        second_cand = sorted_candidates[1] if len(sorted_candidates) > 1 else None

        winner_type = DocumentType.UNKNOWN
        confidence = 0.0
        is_ambiguous = False
        ambiguity_reason = None

        if (
            top_cand
            and top_cand.raw_score >= self.config.min_score
            and top_cand.evidence_count >= self.config.min_evidence_count
        ):
            # Monotonic bounded confidence formula: score / (score + 2.5)
            raw_conf = top_cand.raw_score / (top_cand.raw_score + 2.5)
            raw_conf = min(1.0, max(0.0, raw_conf))

            # Check for ambiguity margin with the runner up
            if second_cand and second_cand.raw_score > 0:
                score_diff = top_cand.raw_score - second_cand.raw_score
                relative_margin = score_diff / (top_cand.raw_score + second_cand.raw_score)

                if relative_margin < self.config.ambiguity_margin and second_cand.raw_score >= self.config.min_score:
                    is_ambiguous = True
                    ambiguity_reason = (
                        f"Margin between {top_cand.document_type.value} ({top_cand.raw_score:.1f}) "
                        f"and {second_cand.document_type.value} ({second_cand.raw_score:.1f}) "
                        f"is within ambiguity threshold ({self.config.ambiguity_margin:.2f})"
                    )

            if raw_conf >= self.config.confidence_threshold and not (
                is_ambiguous and raw_conf < self.config.confidence_threshold + 0.10
            ):
                winner_type = top_cand.document_type
                confidence = raw_conf
            elif is_ambiguous:
                # Ambiguity reduced confidence below usable certainty
                winner_type = DocumentType.UNKNOWN
                confidence = 0.0
            else:
                winner_type = DocumentType.UNKNOWN
                confidence = 0.0
        else:
            winner_type = DocumentType.UNKNOWN
            confidence = 0.0

        # Calculate Normalized Relative Scores Across Candidates
        total_raw = sum(c.raw_score for c in candidates.values())
        candidate_scores: Dict[str, float] = {}
        for doc_type_str, cand in candidates.items():
            if total_raw > 0:
                norm = cand.raw_score / total_raw
            else:
                norm = 0.0
            cand.normalized_score = norm
            candidate_scores[doc_type_str] = norm

        elapsed = time.perf_counter() - start_time

        result = ClassificationResult(
            document_type=winner_type,
            confidence=confidence,
            method="rule_based",
            version="1.0",
            candidate_scores=candidate_scores,
            candidates=candidates,
            evidence=all_evidence,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            execution_time_seconds=elapsed,
            metadata={
                "min_score": self.config.min_score,
                "confidence_threshold": self.config.confidence_threshold,
                "ambiguity_margin": self.config.ambiguity_margin,
                "total_evidence_count": len(all_evidence),
            },
        )

        # Mutate document state
        document.classified_type = result.document_type
        document.classification_confidence = result.confidence
        document.classification_method = result.method

        logger.info(
            f"Classified document '{document.id}' as {result.document_type.value.upper()} "
            f"(confidence={result.confidence:.2f}, evidence={len(all_evidence)}, time={elapsed*1000:.1f}ms)"
        )

        return result
