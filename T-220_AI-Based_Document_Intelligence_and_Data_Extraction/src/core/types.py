"""Domain enumerations and type definitions for DocuMind AI."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class DocumentType(str, Enum):
    """Categorization of ingested documents."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    FORM = "form"
    RESUME = "resume"
    GENERAL_DOCUMENT = "general_document"
    UNKNOWN = "unknown"


class FieldType(str, Enum):
    """Semantic data type for extracted key-value fields."""
    STRING = "string"
    DATE = "date"
    CURRENCY = "currency"
    NUMBER = "number"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    ORGANIZATION = "organization"
    ADDRESS = "address"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    LIST = "list"
    TABLE = "table"


class SeverityLevel(str, Enum):
    """Severity classification for validation findings and review flags."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Verification outcome of deterministic or heuristic rule evaluation."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    SKIPPED = "skipped"
    UNVALIDATED = "unvalidated"


class ConfidenceSource(str, Enum):
    """Origin methodology of a reported confidence score."""
    HEURISTIC = "heuristic"
    RULE_SCORE = "rule_score"
    MODEL_PROBABILITY = "model_probability"
    OCR_PROPAGATED = "ocr_propagated"
    COMPOSITE = "composite"


class ExtractionMethod(str, Enum):
    """Technical mechanism used to locate and parse an extracted entity."""
    REGEX_PATTERN = "regex_pattern"
    KEY_VALUE_HEURISTIC = "key_value_heuristic"
    TABLE_STRUCTURE_PARSER = "table_structure_parser"
    TEMPLATE_MATCH = "template_match"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    LAYOUT_ANALYSIS = "layout_analysis"
    MANUAL_INPUT = "manual_input"
    MOCK = "mock"


class ReviewTriggerType(str, Enum):
    """Specific condition triggering human review requirements."""
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    LOW_EXTRACTION_CONFIDENCE = "low_extraction_confidence"
    VALIDATION_FAILURE = "validation_failure"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    ARITHMETIC_MISMATCH = "arithmetic_mismatch"
    SUSPICIOUS_FORMAT = "suspicious_format"
    UNRECOGNIZED_DOCUMENT_TYPE = "unrecognized_document_type"
