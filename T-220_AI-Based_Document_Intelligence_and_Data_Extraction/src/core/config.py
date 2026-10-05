"""Configuration management for DocuMind AI."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class IngestionConfig:
    """Settings for document intake, format validation, and PDF rasterization."""
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    )
    supported_mime_types: List[str] = field(
        default_factory=lambda: [
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/bmp",
        ]
    )
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    render_dpi: int = 300
    pdf_rendering_timeout_seconds: int = 60
    poppler_path: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """Settings for computer vision image preprocessing."""
    enabled: bool = True
    apply_grayscale: bool = True
    resize_max_dimension: Optional[int] = 2500
    target_width: Optional[int] = None
    apply_denoising: bool = True
    denoise_h: float = 10.0
    apply_contrast_enhancement: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    apply_binarization: bool = False
    binarization_method: str = "otsu"  # otsu, adaptive_gaussian, adaptive_mean
    apply_deskew: bool = True
    deskew_min_angle: float = 0.5
    deskew_max_angle: float = 45.0
    apply_border_cleanup: bool = False
    border_cleanup_fraction: float = 0.01
    debug_save_intermediate_steps: bool = False


@dataclass
class QualityConfig:
    """Settings for heuristic image quality checks and diagnostic warnings."""
    enabled: bool = True
    blur_threshold: float = 100.0       # Variance of Laplacian below this triggers blur warning
    min_brightness: float = 40.0        # Mean intensity below this triggers under-exposed warning
    max_brightness: float = 235.0       # Mean intensity above this triggers over-exposed warning
    min_contrast: float = 30.0          # Intensity std-dev below this triggers low contrast warning
    min_width: int = 600                # Minimum pixel width for OCR reliability
    min_height: int = 600               # Minimum pixel height for OCR reliability


@dataclass
class OCRConfig:
    """Settings for the optical character recognition backend."""
    engine_type: str = "tesseract"  # tesseract, mock, easyocr, paddleocr
    languages: List[str] = field(default_factory=lambda: ["eng"])
    page_segmentation_mode: int = 3  # PSM 3: Fully automatic page segmentation
    ocr_engine_mode: int = 3         # OEM 3: Default, based on what is available
    min_confidence_threshold: float = 0.0  # Flag tokens below this confidence
    tesseract_cmd: Optional[str] = None
    preserve_interword_spaces: bool = False


@dataclass
class ClassificationConfig:
    """Settings for document type classification."""
    default_type: str = "unknown"
    confidence_threshold: float = 0.60
    min_score: float = 2.0
    min_evidence_count: int = 1
    ambiguity_margin: float = 0.10
    keyword_weights_path: Optional[str] = None
    header_weight_multiplier: float = 1.25
    custom_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionSpatialConfig:
    """Spatial proximity thresholds for label-value pairing."""
    same_line_tolerance_px: float = 12.0
    max_horizontal_gap_px: float = 350.0
    max_vertical_gap_px: float = 60.0
    align_tolerance_px: float = 20.0


@dataclass
class ExtractionNormalizationConfig:
    """Normalization options for extracted values."""
    normalize_currency: bool = True
    normalize_dates: bool = True
    default_date_order: str = "DMY"
    strip_whitespace: bool = True
    normalize_phone_numbers: bool = True


@dataclass
class ExtractionEntitiesConfig:
    """Flags controlling generic entity extraction categories."""
    extract_dates: bool = True
    extract_money: bool = True
    extract_email: bool = True
    extract_phone: bool = True
    extract_identifiers: bool = True
    extract_organizations: bool = True
    extract_addresses: bool = True
    extract_persons: bool = True


@dataclass
class ExtractionConfig:
    """Settings for key-value, entity, and table extraction."""
    enabled: bool = True
    extractor_type: str = "rule_based"  # rule_based, ml, hybrid
    confidence_threshold: float = 0.50
    enable_regex_extraction: bool = True
    enable_kv_heuristics: bool = True
    enable_spatial_matching: bool = True
    enable_table_extraction: bool = False
    spatial: ExtractionSpatialConfig = field(default_factory=ExtractionSpatialConfig)
    normalization: ExtractionNormalizationConfig = field(default_factory=ExtractionNormalizationConfig)
    entities: ExtractionEntitiesConfig = field(default_factory=ExtractionEntitiesConfig)
    invoice_required_fields: List[str] = field(
        default_factory=lambda: [
            "invoice_number",
            "invoice_date",
            "vendor_name",
            "total",
        ]
    )
    receipt_required_fields: List[str] = field(
        default_factory=lambda: [
            "merchant_name",
            "transaction_date",
            "total",
        ]
    )
    form_required_fields: List[str] = field(
        default_factory=lambda: [
            "applicant_name",
            "date_of_birth",
        ]
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionConfig:
        """Construct ExtractionConfig from dictionary."""
        spatial_data = data.get("spatial", {})
        norm_data = data.get("normalization", {})
        entities_data = data.get("entities", {})

        return cls(
            enabled=bool(data.get("enabled", True)),
            extractor_type=data.get("extractor_type", "rule_based"),
            confidence_threshold=float(data.get("confidence_threshold", 0.50)),
            enable_regex_extraction=bool(data.get("enable_regex_extraction", True)),
            enable_kv_heuristics=bool(data.get("enable_kv_heuristics", True)),
            enable_spatial_matching=bool(data.get("enable_spatial_matching", True)),
            enable_table_extraction=bool(data.get("enable_table_extraction", False)),
            spatial=ExtractionSpatialConfig(**spatial_data) if spatial_data else ExtractionSpatialConfig(),
            normalization=ExtractionNormalizationConfig(**norm_data) if norm_data else ExtractionNormalizationConfig(),
            entities=ExtractionEntitiesConfig(**entities_data) if entities_data else ExtractionEntitiesConfig(),
            invoice_required_fields=list(data.get("invoice_required_fields", ["invoice_number", "invoice_date", "vendor_name", "total"])),
            receipt_required_fields=list(data.get("receipt_required_fields", ["merchant_name", "transaction_date", "total"])),
            form_required_fields=list(data.get("form_required_fields", ["applicant_name", "date_of_birth"])),
        )



@dataclass
class TableDetectionConfig:
    """Settings for discovering 2D table boundaries."""
    min_rows: int = 2
    min_columns: int = 2
    max_row_gap_px: float = 60.0
    enable_headerless_detection: bool = True


@dataclass
class TableReconstructionConfig:
    """Settings for grid reconstruction, column alignment, and row clustering."""
    row_tolerance_px: float = 12.0
    column_gap_threshold_px: float = 15.0
    alignment_tolerance_px: float = 10.0
    max_multiline_gap_factor: float = 1.6


@dataclass
class TableValidationConfig:
    """Settings for arithmetic row checks and table sum verification."""
    amount_tolerance: float = 0.02
    total_tolerance: float = 0.05
    subtotal_tolerance: float = 0.05


@dataclass
class TableContinuationConfig:
    """Settings for multi-page table continuation linking."""
    enabled: bool = True
    min_column_overlap_ratio: float = 0.65
    max_column_count_diff: int = 0


@dataclass
class TableConfig:
    """Master configuration aggregate for table intelligence and line-item extraction."""
    enabled: bool = True
    detection: TableDetectionConfig = field(default_factory=TableDetectionConfig)
    reconstruction: TableReconstructionConfig = field(default_factory=TableReconstructionConfig)
    validation: TableValidationConfig = field(default_factory=TableValidationConfig)
    continuation: TableContinuationConfig = field(default_factory=TableContinuationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableConfig:
        """Construct TableConfig from dictionary."""
        det_data = data.get("detection", {})
        rec_data = data.get("reconstruction", {})
        val_data = data.get("validation", {})
        cont_data = data.get("continuation", {})

        return cls(
            enabled=bool(data.get("enabled", True)),
            detection=TableDetectionConfig(**det_data) if det_data else TableDetectionConfig(),
            reconstruction=TableReconstructionConfig(**rec_data) if rec_data else TableReconstructionConfig(),
            validation=TableValidationConfig(**val_data) if val_data else TableValidationConfig(),
            continuation=TableContinuationConfig(**cont_data) if cont_data else TableContinuationConfig(),
        )


@dataclass
class ValidationConfig:
    """Settings for deterministic and arithmetic validation rules."""
    strict_mode: bool = False
    date_formats: List[str] = field(
        default_factory=lambda: [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %b %Y",
            "%d %B %Y",
        ]
    )
    currency_symbols: List[str] = field(
        default_factory=lambda: ["$", "€", "£", "₹", "¥", "USD", "EUR", "GBP", "INR"]
    )
    subtotal_tax_tolerance: float = 0.05  # Allow minor rounding differences ($0.05)
    line_item_amount_tolerance: float = 0.02
    table_subtotal_tolerance: float = 0.05
    min_valid_year: int = 1900
    max_valid_year: int = 2100
    invoice_required_fields: List[str] = field(
        default_factory=lambda: ["invoice_number", "invoice_date", "vendor_name", "total"]
    )
    receipt_required_fields: List[str] = field(
        default_factory=lambda: ["merchant_name", "transaction_date", "total"]
    )
    form_required_fields: List[str] = field(
        default_factory=lambda: ["applicant_name", "date_of_birth"]
    )
    enable_required_field_validation: bool = True
    enable_format_validation: bool = True
    enable_cross_field_validation: bool = True
    enable_arithmetic_validation: bool = True
    enable_conflict_detection: bool = True
    enable_table_validation: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationConfig:
        """Construct ValidationConfig from dictionary."""
        return cls(
            strict_mode=bool(data.get("strict_mode", False)),
            date_formats=list(data.get("date_formats", [
                "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
                "%B %d, %Y", "%b %d, %Y", "%d %b %Y", "%d %B %Y",
            ])),
            currency_symbols=list(data.get("currency_symbols", ["$", "€", "£", "₹", "¥", "USD", "EUR", "GBP", "INR"])),
            subtotal_tax_tolerance=float(data.get("subtotal_tax_tolerance", 0.05)),
            line_item_amount_tolerance=float(data.get("line_item_amount_tolerance", 0.02)),
            table_subtotal_tolerance=float(data.get("table_subtotal_tolerance", 0.05)),
            min_valid_year=int(data.get("min_valid_year", 1900)),
            max_valid_year=int(data.get("max_valid_year", 2100)),
            invoice_required_fields=list(data.get("invoice_required_fields", ["invoice_number", "invoice_date", "vendor_name", "total"])),
            receipt_required_fields=list(data.get("receipt_required_fields", ["merchant_name", "transaction_date", "total"])),
            form_required_fields=list(data.get("form_required_fields", ["applicant_name", "date_of_birth"])),
            enable_required_field_validation=bool(data.get("enable_required_field_validation", True)),
            enable_format_validation=bool(data.get("enable_format_validation", True)),
            enable_cross_field_validation=bool(data.get("enable_cross_field_validation", True)),
            enable_arithmetic_validation=bool(data.get("enable_arithmetic_validation", True)),
            enable_conflict_detection=bool(data.get("enable_conflict_detection", True)),
            enable_table_validation=bool(data.get("enable_table_validation", True)),
        )


@dataclass
class FieldConfidenceConfig:
    """Weights and thresholds for field-level confidence calculation."""
    extraction_weight: float = 0.45
    ocr_weight: float = 0.35
    validation_weight: float = 0.20
    conflict_penalty: float = 0.15
    high_threshold: float = 0.85
    medium_threshold: float = 0.65
    low_threshold: float = 0.40
    min_confidence_threshold: float = 0.60


@dataclass
class TableConfidenceConfig:
    """Weights and thresholds for table and line-item confidence calculation."""
    structure_weight: float = 0.30
    row_weight: float = 0.30
    cell_weight: float = 0.20
    validation_weight: float = 0.20
    high_threshold: float = 0.85
    medium_threshold: float = 0.65
    low_threshold: float = 0.40
    min_line_item_threshold: float = 0.60


@dataclass
class DocumentConfidenceConfig:
    """Weights and thresholds for document-level composite confidence calculation."""
    classification_weight: float = 0.20
    field_weight: float = 0.40
    table_weight: float = 0.20
    validation_weight: float = 0.20
    high_threshold: float = 0.85
    medium_threshold: float = 0.65
    low_threshold: float = 0.40
    review_confidence_threshold: float = 0.70


@dataclass
class ReviewRoutingConfig:
    """Settings for human-in-the-loop review triggering."""
    enabled: bool = True
    low_confidence_threshold: float = 0.70
    critical_validation_required: bool = True
    missing_required_field: bool = True
    conflict_detection: bool = True
    table_arithmetic_mismatch: bool = True
    low_ocr_threshold: float = 0.60
    low_extraction_threshold: float = 0.50
    max_low_confidence_fields_for_auto_approval: int = 0


@dataclass
class ReviewPriorityConfig:
    """Priority mapping configurations for review queue routing."""
    critical: str = "urgent"
    high: str = "high"
    medium: str = "medium"
    low: str = "low"


@dataclass
class ConfidenceConfig:
    """Settings for computing composite confidence and human review routing."""
    enabled: bool = True
    ocr_weight: float = 0.35
    extraction_weight: float = 0.40
    validation_weight: float = 0.25
    review_confidence_threshold: float = 0.70  # Documents below 0.70 require review
    flag_missing_required_fields: bool = True
    field: FieldConfidenceConfig = dc_field(default_factory=FieldConfidenceConfig)
    table: TableConfidenceConfig = dc_field(default_factory=TableConfidenceConfig)
    document: DocumentConfidenceConfig = dc_field(default_factory=DocumentConfidenceConfig)
    review: ReviewRoutingConfig = dc_field(default_factory=ReviewRoutingConfig)
    priority: ReviewPriorityConfig = dc_field(default_factory=ReviewPriorityConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConfidenceConfig:
        """Construct ConfidenceConfig from dictionary."""
        fld_data = data.get("field", {})
        tbl_data = data.get("table", {})
        doc_data = data.get("document", {})
        rev_data = data.get("review", {})
        prio_data = data.get("priority", {})

        return cls(
            enabled=bool(data.get("enabled", True)),
            ocr_weight=float(data.get("ocr_weight", 0.35)),
            extraction_weight=float(data.get("extraction_weight", 0.40)),
            validation_weight=float(data.get("validation_weight", 0.25)),
            review_confidence_threshold=float(data.get("review_confidence_threshold", 0.70)),
            flag_missing_required_fields=bool(data.get("flag_missing_required_fields", True)),
            field=FieldConfidenceConfig(**fld_data) if fld_data else FieldConfidenceConfig(),
            table=TableConfidenceConfig(**tbl_data) if tbl_data else TableConfidenceConfig(),
            document=DocumentConfidenceConfig(**doc_data) if doc_data else DocumentConfidenceConfig(),
            review=ReviewRoutingConfig(**rev_data) if rev_data else ReviewRoutingConfig(),
            priority=ReviewPriorityConfig(**prio_data) if prio_data else ReviewPriorityConfig(),
        )


@dataclass
class VisualizationOutputConfig:
    """Settings for visualization file output destinations and serialization."""
    directory: str = "outputs/visualizations"
    format: str = "png"
    save_manifest: bool = True
    save_summary: bool = True
    overwrite_existing: bool = True


@dataclass
class VisualizationLayersConfig:
    """Toggle individual visual annotation layers."""
    ocr: bool = True
    fields: bool = True
    entities: bool = True
    tables: bool = True
    validation: bool = True
    confidence: bool = True
    review: bool = True


@dataclass
class VisualizationRenderingConfig:
    """Styling, typography, transparency, and labeling limits for rendered overlays."""
    line_width: int = 2
    font_size: int = 12
    show_labels: bool = True
    show_confidence: bool = True
    show_review_reasons: bool = True
    max_labels_per_page: int = 300
    opacity: float = 0.85
    draw_legend: bool = True


@dataclass
class VisualizationCoordinatesConfig:
    """Settings for geometric transformations and boundary clipping."""
    clip_to_page: bool = True
    require_valid_dimensions: bool = True
    reject_unknown_coordinate_system: bool = True


@dataclass
class VisualizationEvidenceConfig:
    """Settings for evidence manifest extraction and traceability."""
    include_source_text: bool = True
    include_provenance: bool = True
    include_unlocated_evidence: bool = True


@dataclass
class VisualizationConfig:
    """Master configuration for Phase 9 Visual Explainability & Evidence Overlays."""
    enabled: bool = True
    output: VisualizationOutputConfig = dc_field(default_factory=VisualizationOutputConfig)
    layers: VisualizationLayersConfig = dc_field(default_factory=VisualizationLayersConfig)
    rendering: VisualizationRenderingConfig = dc_field(default_factory=VisualizationRenderingConfig)
    coordinates: VisualizationCoordinatesConfig = dc_field(default_factory=VisualizationCoordinatesConfig)
    evidence: VisualizationEvidenceConfig = dc_field(default_factory=VisualizationEvidenceConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisualizationConfig:
        """Construct VisualizationConfig from dictionary."""
        out_data = data.get("output", {})
        layers_data = data.get("layers", {})
        rend_data = data.get("rendering", {})
        coord_data = data.get("coordinates", {})
        ev_data = data.get("evidence", {})

        return cls(
            enabled=bool(data.get("enabled", True)),
            output=VisualizationOutputConfig(**out_data) if out_data else VisualizationOutputConfig(),
            layers=VisualizationLayersConfig(**layers_data) if layers_data else VisualizationLayersConfig(),
            rendering=VisualizationRenderingConfig(**rend_data) if rend_data else VisualizationRenderingConfig(),
            coordinates=VisualizationCoordinatesConfig(**coord_data) if coord_data else VisualizationCoordinatesConfig(),
            evidence=VisualizationEvidenceConfig(**ev_data) if ev_data else VisualizationEvidenceConfig(),
        )


@dataclass
class StorageConfig:
    """Directory paths for inputs, outputs, and intermediate data."""
    base_dir: str = "."
    input_dir: str = "data/input"
    samples_dir: str = "data/samples"
    processed_dir: str = "data/processed"
    output_json_dir: str = "outputs/json"
    output_csv_dir: str = "outputs/csv"
    output_reports_dir: str = "outputs/reports"
    output_visualizations_dir: str = "outputs/visualizations"


@dataclass
class DocuMindConfig:
    """Master configuration aggregate for DocuMind AI."""
    project_name: str = "DocuMind AI"
    version: str = "0.1.0"
    environment: str = "development"
    ingestion: IngestionConfig = dc_field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = dc_field(default_factory=PreprocessingConfig)
    quality: QualityConfig = dc_field(default_factory=QualityConfig)
    ocr: OCRConfig = dc_field(default_factory=OCRConfig)
    classification: ClassificationConfig = dc_field(default_factory=ClassificationConfig)
    extraction: ExtractionConfig = dc_field(default_factory=ExtractionConfig)
    tables: TableConfig = dc_field(default_factory=TableConfig)
    validation: ValidationConfig = dc_field(default_factory=ValidationConfig)
    confidence: ConfidenceConfig = dc_field(default_factory=ConfidenceConfig)
    visualization: VisualizationConfig = dc_field(default_factory=VisualizationConfig)
    storage: StorageConfig = dc_field(default_factory=StorageConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize configuration to a formatted YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocuMindConfig:
        """Construct configuration from dictionary with nested sub-models."""
        ingestion_data = data.get("ingestion", {})
        preprocessing_data = data.get("preprocessing", {})
        quality_data = data.get("quality", {})
        ocr_data = data.get("ocr", {})
        classification_data = data.get("classification", {})
        extraction_data = data.get("extraction", {})
        tables_data = data.get("tables", data.get("table", {}))
        validation_data = data.get("validation", {})
        confidence_data = data.get("confidence", {})
        visualization_data = data.get("visualization", {})
        storage_data = data.get("storage", {})

        return cls(
            project_name=data.get("project_name", "DocuMind AI"),
            version=data.get("version", "0.1.0"),
            environment=data.get("environment", "development"),
            ingestion=IngestionConfig(**ingestion_data) if ingestion_data else IngestionConfig(),
            preprocessing=PreprocessingConfig(**preprocessing_data) if preprocessing_data else PreprocessingConfig(),
            quality=QualityConfig(**quality_data) if quality_data else QualityConfig(),
            ocr=OCRConfig(**ocr_data) if ocr_data else OCRConfig(),
            classification=ClassificationConfig(**classification_data) if classification_data else ClassificationConfig(),
            extraction=ExtractionConfig.from_dict(extraction_data) if extraction_data else ExtractionConfig(),
            tables=TableConfig.from_dict(tables_data) if tables_data else TableConfig(),
            validation=ValidationConfig.from_dict(validation_data) if validation_data else ValidationConfig(),
            confidence=ConfidenceConfig.from_dict(confidence_data) if confidence_data else ConfidenceConfig(),
            visualization=VisualizationConfig.from_dict(visualization_data) if visualization_data else VisualizationConfig(),
            storage=StorageConfig(**storage_data) if storage_data else StorageConfig(),
        )


def load_config(config_path: Optional[Union[str, Path]] = None) -> DocuMindConfig:
    """Load configuration from YAML file with fallback to defaults.

    Args:
        config_path: Path to the YAML configuration file.
    """
    if config_path is None:
        default_yaml_path = Path("configs/default.yaml")
        if default_yaml_path.exists():
            config_path = default_yaml_path
        else:
            return DocuMindConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return DocuMindConfig.from_dict(data)
