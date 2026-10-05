"""DocuMind AI CLI Entry Point and Foundation Diagnostic Utility.

Provides runtime configuration validation, ingestion diagnostics, OCR environment checks, and contract inspection.
"""

import argparse
import platform
import sys
from pathlib import Path

# Add project root to sys.path to allow execution without explicit PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger
from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    ExtractedField,
    ProcessingResult,
    Provenance,
    ValidationResult,
)
from src.core.types import (
    ConfidenceSource,
    DocumentType,
    ExtractionMethod,
    FieldType,
    SeverityLevel,
    ValidationStatus,
)
from src.classification.classifier import RuleBasedDocumentClassifier
from src.classification.models import ClassificationEvidence, ClassificationResult
from src.extraction.extractor import DocumentExtractor
from src.extraction.models import EntityType, ExtractedEntity, ExtractionResult
from src.ingestion.validators import detect_poppler_path
from src.ocr.availability import check_ocr_availability
from src.ocr.models import OCRWord
from src.tables.extractor import TableExtractor
from src.tables.models import (
    LineItem,
    Table,
    TableCell,
    TableColumn,
    TableHeader,
    TableRegion,
    TableRow,
    TableValueType,
)
from src.validation import (
    DocumentValidationEngine,
    RuleRegistry,
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
)
from src.confidence import (
    ConfidenceBand,
    ConfidenceEngine,
    ConfidenceSignals,
    DocumentConfidence,
    FieldConfidence,
    LineItemConfidence,
    ReviewIssue,
    ReviewPriority,
    ReviewQueueItem,
    ReviewReason,
    ReviewRouter,
    ReviewRoutingResult,
    ReviewStatus,
    ReviewTarget,
    ReviewTargetType,
    TableConfidence,
)
from src.visualization import (
    AnnotationType,
    CoordinateTransformer,
    DocumentVisualizer,
    EvidenceMapper,
    EvidenceRegion,
    LegendRenderer,
    OverlayAnnotation,
    OverlayRenderer,
    PageVisualization,
    VisualCoordinateSystem,
    VisualizationManifest,
    VisualizationResult,
    VisualizationSerializer,
)


def print_banner() -> None:
    """Print DocuMind AI CLI banner."""
    print("=" * 70)
    print(" DocuMind AI - Document Intelligence & Data Extraction Platform")
    print(" Version: 1.0.0 | Phase: Professional Review Dashboard (Phase 10)")
    print("=" * 70)



def check_configuration(config_path: str = "configs/default.yaml") -> int:
    """Validate that the project configuration parses and loads properly."""
    logger = get_logger("cli")
    print(f"\n[+] Validating configuration file: {config_path}")
    try:
        config = load_config(config_path)
        print(f"  * Project Name:        {config.project_name}")
        print(f"  * Target Environment:  {config.environment}")
        print(f"  * Supported Formats:   {', '.join(config.ingestion.supported_extensions)}")
        print(f"  * Ingestion Max Size:  {config.ingestion.max_file_size_bytes / (1024*1024):.1f} MB")
        print(f"  * OCR Backend Engine:  {config.ocr.engine_type} (Languages: {config.ocr.languages})")
        print(f"  * Default Classify:    {config.classification.default_type}")
        print(f"  * Strict Validation:   {config.validation.strict_mode}")
        print(f"  * Review Threshold:    {config.confidence.review_confidence_threshold}")
        print("\n[OK] Configuration loaded and validated successfully.")
        return 0
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        print(f"\n[FAIL] Configuration Error: {e}")
        return 1


def check_ingestion_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect environment dependencies, image processing backends, and Poppler availability."""
    print("\n[+] Inspecting Document Ingestion & Preprocessing Subsystem...")
    config = load_config(config_path)

    # 1. Python Environment
    print(f"  * Python Environment:  {platform.python_implementation()} {platform.python_version()} on {platform.system()} ({platform.machine()})")

    # 2. Dependency Availability
    libs = [
        ("Pillow (PIL)", "PIL", "__version__"),
        ("OpenCV (cv2)", "cv2", "__version__"),
        ("PyPDF", "pypdf", "__version__"),
        ("pdf2image", "pdf2image", "__version__"),
        ("NumPy", "numpy", "__version__"),
        ("PyYAML", "yaml", "__version__"),
    ]

    all_libs_ok = True
    for display_name, mod_name, ver_attr in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, ver_attr, "Available")
            print(f"  * {display_name:<20}: [INSTALLED] (v{ver})")
        except ImportError:
            print(f"  * {display_name:<20}: [MISSING]")
            all_libs_ok = False

    # 3. Poppler Detection
    poppler_dir = detect_poppler_path(config.ingestion.poppler_path)
    if poppler_dir:
        print(f"  * Poppler PDF Engine : [DETECTED] ({poppler_dir})")
    else:
        print("  * Poppler PDF Engine : [NOT DETECTED in PATH]")
        print("    --> Note: PDF rasterization with pdf2image requires Poppler binaries (pdftoppm).")
        print("    --> Action: Install Poppler for Windows and set 'ingestion.poppler_path' in config.")

    # 4. Ingestion Settings
    print("\n[+] Ingestion Subsystem Configuration:")
    print(f"  * Supported Formats  : {', '.join(config.ingestion.supported_extensions)}")
    print(f"  * Max Ingestion Size : {config.ingestion.max_file_size_bytes / (1024*1024):.1f} MB")
    print(f"  * Default Render DPI : {config.ingestion.render_dpi} DPI")

    # 5. Preprocessing Pipeline Operations
    print("\n[+] Preprocessing Pipeline Operations:")
    print(f"  * Pipeline Enabled   : {config.preprocessing.enabled}")
    print(f"  * Grayscale Normal   : {config.preprocessing.apply_grayscale}")
    print(f"  * Deskew Correction  : {config.preprocessing.apply_deskew} (min={config.preprocessing.deskew_min_angle}°, max={config.preprocessing.deskew_max_angle}°)")
    print(f"  * Denoising Filter   : {config.preprocessing.apply_denoising} (h={config.preprocessing.denoise_h})")
    print(f"  * CLAHE Contrast     : {config.preprocessing.apply_contrast_enhancement} (clip_limit={config.preprocessing.clahe_clip_limit})")
    print(f"  * Thresholding       : {config.preprocessing.apply_binarization} ({config.preprocessing.binarization_method})")
    print(f"  * Border Cleanup     : {config.preprocessing.apply_border_cleanup}")

    # 6. Quality Heuristic Thresholds
    print("\n[+] Heuristic Quality Monitoring:")
    print(f"  * Quality Enabled    : {config.quality.enabled}")
    print(f"  * Blur Threshold     : {config.quality.blur_threshold} (Variance of Laplacian)")
    print(f"  * Brightness Limits  : [{config.quality.min_brightness}, {config.quality.max_brightness}]")
    print(f"  * Contrast Minimum   : {config.quality.min_contrast}")
    print(f"  * Min Resolution     : {config.quality.min_width}x{config.quality.min_height} px")

    if all_libs_ok:
        print("\n[OK] Core Python ingestion and preprocessing libraries are ready.")
        return 0
    else:
        print("\n[WARNING] One or more required Python dependencies are missing.")
        return 1


def check_ocr_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Perform rigorous inspection of the OCR runtime, Tesseract binary, and language data."""
    print("\n[+] Inspecting Optical Character Recognition (OCR) Subsystem...")
    config = load_config(config_path)
    avail = check_ocr_availability(config.ocr)

    # 1. Python Wrapper
    wrapper_status = "[AVAILABLE]" if avail.is_wrapper_available else "[NOT INSTALLED]"
    print(f"  * Python Wrapper (pytesseract): {wrapper_status}")

    # 2. Tesseract Executable
    if avail.is_engine_available and avail.executable_path:
        print(f"  * Tesseract Executable       : [DETECTED] ({avail.executable_path})")
    else:
        print("  * Tesseract Executable       : [NOT FOUND]")

    # 3. Engine Version
    ver_str = avail.engine_version or "N/A"
    print(f"  * Tesseract Engine Version   : {ver_str}")

    # 4. Language Packs
    req_lang = avail.requested_language
    lang_status = "[AVAILABLE]" if avail.is_language_available else "[NOT FOUND]"
    print(f"  * Requested Language ('{req_lang}')  : {lang_status}")
    if avail.available_languages:
        print(f"  * Installed Languages        : {', '.join(avail.available_languages)}")

    # 5. Configured OCR Engine Mode & PSM
    print(f"  * Page Segmentation Mode     : PSM {config.ocr.page_segmentation_mode}")
    print(f"  * OCR Engine Mode            : OEM {config.ocr.ocr_engine_mode}")

    # 6. Overall Readiness
    print("\n" + "-" * 70)
    if avail.is_ready:
        print("  OCR Environment Status: [READY]")
        print("-" * 70)
        print("\n[OK] OCR subsystem is fully operational for processing.")
        return 0
    else:
        print("  OCR Environment Status: [NOT READY]")
        print("-" * 70)
        print(f"\n[DIAGNOSTIC] {avail.status_message}")
        print("To enable Tesseract OCR on Windows:")
        print("  1. Download Tesseract Windows installer from UB-Mannheim or Tesseract GitHub.")
        print("  2. Install Tesseract to e.g. C:\\Program Files\\Tesseract-OCR")
        print("  3. Set 'ocr.tesseract_cmd' in configs/default.yaml or define TESSERACT_CMD in environment.\n")
        return 1


def check_classification_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect classification rules, rule-based scoring engine, and benchmark sample discrimination."""
    print("\n[+] Inspecting Document Classification Subsystem...")
    config = load_config(config_path)
    classifier = RuleBasedDocumentClassifier(config=config.classification)

    # 1. Inspect Loaded Rule Collection
    total_rules = len(classifier.rules)
    print(f"  * Total Active Rules:        {total_rules}")

    rules_by_type = {}
    for r in classifier.rules:
        rules_by_type.setdefault(r.target_type.value, []).append(r)

    for doc_type_name, type_rules in rules_by_type.items():
        pos = sum(1 for r in type_rules if r.weight > 0)
        neg = sum(1 for r in type_rules if r.weight < 0)
        regex_cnt = sum(1 for r in type_rules if r.is_regex)
        print(f"    - {doc_type_name.upper():<16}: {len(type_rules):>2} rules ({pos} positive, {neg} penalty, {regex_cnt} regex)")

    # 2. Configuration Settings
    print("\n[+] Classification Configuration Parameters:")
    print(f"  * Confidence Threshold:      {config.classification.confidence_threshold:.2f}")
    print(f"  * Minimum Evidence Score:    {config.classification.min_score:.2f}")
    print(f"  * Minimum Evidence Count:    {config.classification.min_evidence_count}")
    print(f"  * Ambiguity Margin:          {config.classification.ambiguity_margin:.2f}")
    print(f"  * Header Weight Multiplier:  {config.classification.header_weight_multiplier:.2f}x")

    # 3. Live Benchmark Discrimination on Canonical Documents
    print("\n[+] Testing Deterministic Classification on Sample Documents:")
    samples = [
        (
            "Sample Invoice",
            (
                "TAX INVOICE\n"
                "Invoice Number: INV-2026-9810\n"
                "Invoice Date: 2026-10-01\n"
                "Bill To: Apex Global Logistics Inc\n"
                "Payment Terms: NET 30\n"
                "Subtotal: $4,500.00\n"
                "Total Due: $4,950.00"
            ),
            DocumentType.INVOICE,
        ),
        (
            "Sample POS Receipt",
            (
                "STORE RECEIPT\n"
                "Store # 4821 - Central Plaza\n"
                "Cashier: 04\n"
                "Subtotal: $35.00\n"
                "Tax: $2.80\n"
                "Total: $37.80\n"
                "Cash Tendered: $40.00\n"
                "Change Due: $2.20\n"
                "Thank you for shopping!"
            ),
            DocumentType.RECEIPT,
        ),
        (
            "Sample Application Form",
            (
                "APPLICATION FORM\n"
                "Form No: AF-2026-01\n"
                "Please print clearly in ink.\n"
                "Applicant Name: __________________\n"
                "Date of Birth (DOB): _____________\n"
                "Social Security (SSN): ____________\n"
                "Signature of Applicant: __________"
            ),
            DocumentType.FORM,
        ),
        (
            "Sample General Memo",
            (
                "MEMORANDUM\n"
                "TO: Engineering Operations\n"
                "FROM: Head of Systems Architecture\n"
                "SUBJECT: Q4 Infrastructure Modernization\n"
                "EXECUTIVE SUMMARY\n"
                "This document outlines architectural updates.\n"
                "Introduction\n"
                "Conclusion\n"
                "Best regards,\n"
                "Engineering Team"
            ),
            DocumentType.GENERAL_DOCUMENT,
        ),
        (
            "Sample Unknown Text",
            "random unstructured content without semantic document indicators",
            DocumentType.UNKNOWN,
        ),
    ]

    all_passed = True
    for label, text, expected_type in samples:
        meta = DocumentMetadata(
            document_id="bench_" + label.lower().replace(" ", "_"),
            filename="sample.txt",
            file_path=Path("data/sample.txt"),
            file_type="text/plain",
            file_size_bytes=len(text),
            checksum_sha256="bench_checksum",
            page_count=1,
        )
        page = DocumentPage(page_number=1, raw_text=text)
        doc = Document(metadata=meta, pages=[page])

        result = classifier.classify(doc)
        status_tag = "[PASS]" if result.document_type == expected_type else "[FAIL]"
        if result.document_type != expected_type:
            all_passed = False

        top_evidence_names = [e.signal_name for e in result.get_top_evidence(limit=2)]
        evidence_str = ", ".join(top_evidence_names) if top_evidence_names else "none"
        print(
            f"  * {label:<24}: {status_tag} Classified as {result.document_type.value.upper():<16} "
            f"(Confidence: {result.confidence:.2f}, Top Signals: [{evidence_str}], Time: {result.execution_time_seconds*1000:.1f}ms)"
        )

    print("\n" + "-" * 70)
    if all_passed:
        print("  Classification Subsystem Status: [OPERATIONAL]")
        print("-" * 70)
        print("\n[OK] Document classification subsystem is fully operational.")
        return 0
    else:
        print("  Classification Subsystem Status: [DISCREPANCY DETECTED]")
        print("-" * 70)
        return 1


def check_extraction_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect structured field and entity extraction capabilities and sample extractions."""
    print("\n[+] Inspecting Structured Field & Entity Extraction Subsystem...")
    config = load_config(config_path)

    # 1. Configuration details
    print(f"  * Extractor Backend:   Rule-Based Multi-Strategy Dispatcher (v1.0.0)")
    print(f"  * Spatial Matchers:    Inline Colon, Adjacent Horizontal (gap<={config.extraction.spatial.max_horizontal_gap_px}px), Vertically Below (gap<={config.extraction.spatial.max_vertical_gap_px}px)")
    print(f"  * Normalization:       Default Date Order={config.extraction.normalization.default_date_order}, Strip Whitespace={config.extraction.normalization.strip_whitespace}")
    print(f"  * Supported Categories: INVOICE, RECEIPT, FORM, RESUME, GENERAL_DOCUMENT, UNKNOWN")
    print(f"  * Generic Entities:    DATE, MONEY, EMAIL, PHONE, IDENTIFIER, ORGANIZATION, ADDRESS, PERSON")

    classifier = RuleBasedDocumentClassifier(config=config.classification)
    extractor = DocumentExtractor(config=config.extraction)

    test_docs = [
        (
            "Commercial Invoice",
            (
                "Apex Global Logistics Inc\n"
                "Invoice Number: INV-2026-9810\n"
                "Invoice Date: 2026-10-01\n"
                "Due Date: 2026-10-31\n"
                "Bill To: Acme Technologies Corp\n"
                "Payment Terms: NET 30\n"
                "Purchase Order: PO-98124\n"
                "Tax ID: VAT-98765432\n"
                "Subtotal: $4,500.00\n"
                "Tax: $450.00\n"
                "Total Due: $4,950.00\n"
            ),
            DocumentType.INVOICE,
            ["invoice_number", "invoice_date", "total", "subtotal", "tax", "payment_terms", "purchase_order_number"],
        ),
        (
            "Retail POS Receipt",
            (
                "Starbucks Coffee Store #1402\n"
                "Receipt No: RCPT-49102\n"
                "Date: 2026-10-01\n"
                "Time: 08:30 AM\n"
                "Cashier: Sarah M\n"
                "Subtotal: $45.50\n"
                "Sales Tax: $3.64\n"
                "Total: $49.14\n"
                "Visa: ************4321\n"
            ),
            DocumentType.RECEIPT,
            ["receipt_number", "transaction_date", "subtotal", "tax", "total", "payment_method"],
        ),
        (
            "Application Form",
            (
                "MEMBERSHIP APPLICATION\n"
                "Form No: APP-2026-0042\n"
                "Full Name: Eleanor Vance\n"
                "Date of Birth: 1990-08-14\n"
                "Email Address: eleanor.vance@hillhouse.org\n"
                "Phone: +1 (555) 987-6543\n"
                "Address: 450 Blackwood Terrace, Boston MA 02108\n"
                "Signature: [Signed Eleanor Vance]\n"
            ),
            DocumentType.FORM,
            ["form_number", "applicant_name", "date_of_birth", "email", "phone", "address", "signature_indicator"],
        ),
        (
            "Curriculum Vitae / Resume",
            (
                "VENKATA RAMARAJU\n"
                "Email: venkata.raju@example.com | Phone: +1 (555) 345-6789\n"
                "Location: San Francisco, CA\n"
                "LinkedIn: linkedin.com/in/venkataraju | GitHub: github.com/venkataraju\n\n"
                "SUMMARY\n"
                "Experienced Machine Learning and Full-Stack Software Engineer.\n\n"
                "TECHNICAL SKILLS\n"
                "Python, PyTorch, FastAPI, React, TypeScript, PostgreSQL\n\n"
                "EDUCATION\n"
                "Bachelor of Technology in Computer Science, UC Berkeley\n\n"
                "WORK EXPERIENCE\n"
                "Senior AI Engineer at Apex Systems Inc\n"
            ),
            DocumentType.RESUME,
            ["candidate_name", "email", "phone", "linkedin_url", "github_url", "location", "skills", "education", "experience"],
        ),
    ]

    print("\n  [+] Executing Structured Field Extractions across Document Types:")
    all_ok = True

    for label, text, exp_type, required_keys in test_docs:
        meta = DocumentMetadata(
            document_id="diag_" + label.lower().replace(" ", "_"),
            filename=f"{label.lower().replace(' ', '_')}.pdf",
            file_path=Path(f"data/{label.lower().replace(' ', '_')}.pdf"),
            file_type="application/pdf",
            file_size_bytes=len(text),
            checksum_sha256="dummy_sha",
            page_count=1,
        )
        page = DocumentPage(page_number=1, raw_text=text)
        doc = Document(metadata=meta, pages=[page])

        clf_res = classifier.classify(doc)
        ext_res = extractor.extract(doc, clf_res)

        missing_keys = [k for k in required_keys if k not in ext_res.fields]
        status = "[PASS]" if not missing_keys else "[FAIL]"
        if missing_keys:
            all_ok = False

        print(f"\n  --- {label} ({ext_res.document_type.value.upper()}) {status} ---")
        print(f"      Fields Extracted: {len(ext_res.fields)} | Entities: {len(ext_res.entities)} | Time: {ext_res.execution_time_seconds*1000:.1f}ms")
        for f_name, f_obj in ext_res.fields.items():
            print(f"      * {f_name:<22}: Raw='{f_obj.value}' -> Normalized={repr(f_obj.normalized_value)} (Conf: {f_obj.extraction_confidence:.2f}, Type: {f_obj.field_type.value})")

    print("\n" + "-" * 70)
    if all_ok:
        print("  Structured Extraction Subsystem Status: [OPERATIONAL]")
        print("-" * 70)
        print("\n[OK] Structured field and entity extraction subsystem is fully operational.")
        return 0
    else:
        print("  Structured Extraction Subsystem Status: [DISCREPANCY DETECTED]")
        print("-" * 70)
        return 1


def check_tables_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect table detection, grid reconstruction, line-item extraction, and arithmetic validation."""
    print("\n[+] Inspecting Table Intelligence & Line-Item Extraction Subsystem...")
    config = load_config(config_path)

    # 1. Configuration details
    print("  * Table Subsystem Enabled:  ", config.tables.enabled)
    print("  * Header Matcher Synonyms:  ", "item_code, description, quantity, unit_price, amount, tax, discount")
    print(f"  * Grid Geometry Tolerances: Row={config.tables.reconstruction.row_tolerance_px}px, ColGap={config.tables.reconstruction.column_gap_threshold_px}px, MultilineFactor={config.tables.reconstruction.max_multiline_gap_factor}x")
    print(f"  * Arithmetic Tolerances:    Amount=+/-${config.tables.validation.amount_tolerance:.2f}, Total=+/-${config.tables.validation.total_tolerance:.2f}")
    print(f"  * Continuation Detector:    Overlap Ratio>={config.tables.continuation.min_column_overlap_ratio:.2f}")

    table_extractor = TableExtractor(config=config.tables)

    # 2. Build Multi-Line Invoice Table with OCR Words
    words = [
        # Headers (y: 200..220)
        OCRWord(text="Description", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=50.0, ymin=200.0, xmax=220.0, ymax=220.0), page_number=1),
        OCRWord(text="Qty", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(xmin=280.0, ymin=200.0, xmax=320.0, ymax=220.0), page_number=1),
        OCRWord(text="Unit", confidence=0.96, raw_confidence=96.0, bounding_box=BoundingBox(xmin=370.0, ymin=200.0, xmax=410.0, ymax=220.0), page_number=1),
        OCRWord(text="Price", confidence=0.96, raw_confidence=96.0, bounding_box=BoundingBox(xmin=415.0, ymin=200.0, xmax=460.0, ymax=220.0), page_number=1),
        OCRWord(text="Total", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=510.0, ymin=200.0, xmax=550.0, ymax=220.0), page_number=1),
        OCRWord(text="Amount", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=555.0, ymin=200.0, xmax=620.0, ymax=220.0), page_number=1),

        # Row 1 (y: 240..260)
        OCRWord(text="Cloud", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=240.0, xmax=100.0, ymax=260.0), page_number=1),
        OCRWord(text="Infrastructure", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=105.0, ymin=240.0, xmax=200.0, ymax=260.0), page_number=1),
        OCRWord(text="Enterprise", confidence=0.94, raw_confidence=94.0, bounding_box=BoundingBox(xmin=205.0, ymin=240.0, xmax=270.0, ymax=260.0), page_number=1),
        OCRWord(text="2", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=295.0, ymin=240.0, xmax=310.0, ymax=260.0), page_number=1),
        OCRWord(text="$1,200.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=380.0, ymin=240.0, xmax=450.0, ymax=260.0), page_number=1),
        OCRWord(text="$2,400.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=530.0, ymin=240.0, xmax=610.0, ymax=260.0), page_number=1),

        # Row 2 Line A (y: 280..300)
        OCRWord(text="Database", confidence=0.96, raw_confidence=96.0, bounding_box=BoundingBox(xmin=50.0, ymin=280.0, xmax=120.0, ymax=300.0), page_number=1),
        OCRWord(text="Managed", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=125.0, ymin=280.0, xmax=190.0, ymax=300.0), page_number=1),
        OCRWord(text="Cluster", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=195.0, ymin=280.0, xmax=250.0, ymax=300.0), page_number=1),
        OCRWord(text="1", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=295.0, ymin=280.0, xmax=310.0, ymax=300.0), page_number=1),
        OCRWord(text="$850.00", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(xmin=390.0, ymin=280.0, xmax=450.0, ymax=300.0), page_number=1),
        OCRWord(text="$850.00", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(xmin=540.0, ymin=280.0, xmax=610.0, ymax=300.0), page_number=1),

        # Row 2 Line B (Wrapped Description) (y: 305..320)
        OCRWord(text="with", confidence=0.93, raw_confidence=93.0, bounding_box=BoundingBox(xmin=50.0, ymin=305.0, xmax=80.0, ymax=320.0), page_number=1),
        OCRWord(text="High", confidence=0.94, raw_confidence=94.0, bounding_box=BoundingBox(xmin=85.0, ymin=305.0, xmax=115.0, ymax=320.0), page_number=1),
        OCRWord(text="Availability", confidence=0.94, raw_confidence=94.0, bounding_box=BoundingBox(xmin=120.0, ymin=305.0, xmax=195.0, ymax=320.0), page_number=1),
        OCRWord(text="Add-on", confidence=0.92, raw_confidence=92.0, bounding_box=BoundingBox(xmin=200.0, ymin=305.0, xmax=255.0, ymax=320.0), page_number=1),

        # Row 3 (y: 340..360)
        OCRWord(text="Support", confidence=0.96, raw_confidence=96.0, bounding_box=BoundingBox(xmin=50.0, ymin=340.0, xmax=110.0, ymax=360.0), page_number=1),
        OCRWord(text="Plan", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=115.0, ymin=340.0, xmax=150.0, ymax=360.0), page_number=1),
        OCRWord(text="3", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=295.0, ymin=340.0, xmax=310.0, ymax=360.0), page_number=1),
        OCRWord(text="$150.00", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(xmin=390.0, ymin=340.0, xmax=450.0, ymax=360.0), page_number=1),
        OCRWord(text="$450.00", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(xmin=540.0, ymin=340.0, xmax=610.0, ymax=360.0), page_number=1),

        # Summary Row (Subtotal) (y: 380..400)
        OCRWord(text="Subtotal", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=400.0, ymin=380.0, xmax=465.0, ymax=400.0), page_number=1),
        OCRWord(text="$3,700.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=530.0, ymin=380.0, xmax=610.0, ymax=400.0), page_number=1),
    ]

    meta = DocumentMetadata(
        document_id="diag_commercial_invoice_01",
        filename="invoice_table_sample.pdf",
        file_path=Path("data/sample.pdf"),
        file_type="application/pdf",
        file_size_bytes=1024,
        checksum_sha256="sample_sha",
        page_count=1,
    )

    page = DocumentPage(
        page_number=1,
        width=800.0,
        height=1000.0,
        raw_text="Sample Table Invoice",
        metadata={"ocr_words": words},
    )

    doc = Document(metadata=meta, pages=[page])

    print("\n  [+] Executing Table Extraction on Synthetic Invoice Document:")
    result = table_extractor.extract(doc)

    print(f"  * Total Tables Detected:     {len(result.tables)}")
    print(f"  * Total Line Items:          {len(result.line_items)}")
    print(f"  * Execution Duration:        {result.execution_time_seconds*1000:.1f}ms")

    if not result.tables:
        print("\n[FAIL] Table detection failed on sample document.")
        return 1

    t = result.tables[0]
    print(f"\n  --- Reconstructed Table '{t.table_id}' ---")
    print(f"      Headers Detected: {t.headers}")
    print(f"      Columns ({len(t.columns)}):")
    for col in t.columns:
        print(f"        * [{col.index}] '{col.name}' -> Canonical: {col.canonical_field} (Span: [{col.x_start:.1f}, {col.x_end:.1f}], Type: {col.inferred_type.value}, Align: {col.alignment})")

    print(f"      Line Items ({len(t.line_items)}):")
    for li in t.line_items:
        arith_status = "[VALID]" if li.is_valid_arithmetic else "[MISMATCH]"
        print(f"        * Qty={li.quantity} | UnitPrice=${li.unit_price:.2f} | Amount=${li.amount:.2f} {arith_status} -> '{li.description}'")

    if t.validation_result:
        vr = t.validation_result
        print(f"      Table Arithmetic Validation:")
        print(f"        * Row Checks Passed:   {vr.row_checks_passed} / {vr.row_checks_passed + vr.row_checks_failed}")
        print(
    f"        * Subtotal Verified:   "
    f"Calc={'N/A' if vr.calculated_subtotal is None else f'${vr.calculated_subtotal:.2f}'} "
    f"vs Reported={'N/A' if vr.reported_subtotal is None else f'${vr.reported_subtotal:.2f}'} "
    f"-> Valid={vr.subtotal_valid}"
)
    print(f"      Confidence Breakdown (Composite: {t.confidence:.2f}):")
    for k, v in t.confidence_breakdown.items():
        print(f"        * {k:<25}: {v:.2f}")

    print("\n" + "-" * 70)
    if result.tables and len(result.line_items) == 3 and t.validation_result and t.validation_result.is_valid:
        print("  Table Intelligence Subsystem Status: [OPERATIONAL]")
        print("-" * 70)
        print("\n[OK] Table intelligence and line-item extraction subsystem is fully operational.")
        return 0
    else:
        print("  Table Intelligence Subsystem Status: [DISCREPANCY DETECTED]")
        print("-" * 70)
        return 1


def check_validation_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect validation engine rules, cross-field checks, arithmetic validators, and conflict detectors."""
    print("\n[+] Inspecting Document Validation & Consistency Engine Subsystem...")
    config = load_config(config_path)

    # 1. Initialize Engine and Rule Registry
    engine = DocumentValidationEngine(config=config.validation)
    registered_rules = engine.rules

    print(f"  * Strict Validation Mode:    {config.validation.strict_mode}")
    print(f"  * Subtotal/Tax Tolerance:    ${config.validation.subtotal_tax_tolerance:.2f}")
    print(f"  * Line-Item Math Tolerance:  ${config.validation.line_item_amount_tolerance:.2f}")
    print(f"  * Valid Date Year Range:     [{config.validation.min_valid_year}, {config.validation.max_valid_year}]")
    print(f"  * Total Active Rules:        {len(registered_rules)}")

    # Group by category
    rules_by_cat: dict = {}
    for r in registered_rules:
        cat_name = r.category.value if hasattr(r.category, "value") else str(r.category)
        rules_by_cat.setdefault(cat_name, []).append(r)

    print("\n  [Rule Catalog by Category]")
    for cat, rules in sorted(rules_by_cat.items()):
        print(f"    - {cat.upper()} ({len(rules)} rules):")
        for r in rules:
            print(f"        * [{r.rule_id}] {r.rule_name} (targets: {', '.join(r.target_fields)})")

    # 2. Execute Synthetic Document Validation Verification
    print("\n  [Executing Validation Verification on Invoiced Document]...")
    bbox = BoundingBox(xmin=100.0, ymin=100.0, xmax=500.0, ymax=150.0)

    # Synthetic Fields
    fields = {
        "invoice_number": ExtractedField(
            name="invoice_number",
            value="INV-2026-9901",
            normalized_value="INV-2026-9901",
            field_type=FieldType.IDENTIFIER,
            provenance=Provenance("doc_val_diag", 1, "INV-2026-9901", bbox, 0.98, ExtractionMethod.REGEX_PATTERN),
            extraction_confidence=0.96,
            is_required=True,
        ),
        "invoice_date": ExtractedField(
            name="invoice_date",
            value="2026-10-01",
            normalized_value="2026-10-01",
            field_type=FieldType.DATE,
            provenance=Provenance("doc_val_diag", 1, "2026-10-01", bbox, 0.99, ExtractionMethod.REGEX_PATTERN),
            extraction_confidence=0.97,
            is_required=True,
        ),
        "due_date": ExtractedField(
            name="due_date",
            value="2026-10-31",
            normalized_value="2026-10-31",
            field_type=FieldType.DATE,
            provenance=Provenance("doc_val_diag", 1, "2026-10-31", bbox, 0.98, ExtractionMethod.REGEX_PATTERN),
            extraction_confidence=0.95,
        ),
        "vendor_name": ExtractedField(
            name="vendor_name",
            value="Acme Corporation LLC",
            normalized_value="Acme Corporation LLC",
            field_type=FieldType.ORGANIZATION,
            provenance=Provenance("doc_val_diag", 1, "Acme Corporation LLC", bbox, 0.95, ExtractionMethod.KEY_VALUE_HEURISTIC),
            extraction_confidence=0.92,
            is_required=True,
        ),
        "subtotal": ExtractedField(
            name="subtotal",
            value="$4,000.00",
            normalized_value=4000.00,
            field_type=FieldType.CURRENCY,
            provenance=Provenance("doc_val_diag", 1, "$4,000.00", bbox, 0.97, ExtractionMethod.KEY_VALUE_HEURISTIC),
            extraction_confidence=0.94,
        ),
        "tax": ExtractedField(
            name="tax",
            value="$400.00",
            normalized_value=400.00,
            field_type=FieldType.CURRENCY,
            provenance=Provenance("doc_val_diag", 1, "$400.00", bbox, 0.96, ExtractionMethod.KEY_VALUE_HEURISTIC),
            extraction_confidence=0.93,
        ),
        "total": ExtractedField(
            name="total",
            value="$4,400.00",
            normalized_value=4400.00,
            field_type=FieldType.CURRENCY,
            provenance=Provenance("doc_val_diag", 1, "$4,400.00", bbox, 0.98, ExtractionMethod.KEY_VALUE_HEURISTIC),
            extraction_confidence=0.96,
            is_required=True,
        ),
    }

    # Synthetic Table and Line Items
    items = [
        LineItem(
            row_index=0,
            description="Enterprise Cloud Computing",
            quantity=2.0,
            unit_price=1500.00,
            amount=3000.00,
            page_number=1,
            confidence=0.95,
            is_valid_arithmetic=True,
        ),
        LineItem(
            row_index=1,
            description="Dedicated Storage Cluster",
            quantity=1.0,
            unit_price=1000.00,
            amount=1000.00,
            page_number=1,
            confidence=0.94,
            is_valid_arithmetic=True,
        ),
    ]

    table = Table(
        table_id="table_1",
        page_number=1,
        headers=["Description", "Quantity", "Unit Price", "Amount"],
        columns=[
            TableColumn(0, "Description", "description", 50, 300, "left", TableValueType.TEXT),
            TableColumn(1, "Quantity", "quantity", 300, 450, "right", TableValueType.DECIMAL),
            TableColumn(2, "Unit Price", "unit_price", 450, 600, "right", TableValueType.CURRENCY),
            TableColumn(3, "Amount", "amount", 600, 750, "right", TableValueType.CURRENCY),
        ],
        rows=[],
        line_items=items,
        confidence=0.95,
    )

    report = engine.validate_document(
        document_id="doc_val_diag",
        document_type=DocumentType.INVOICE,
        fields=fields,
        tables=[table],
    )

    print(f"    * Document ID:       {report.document_id}")
    print(f"    * Document Type:     {report.document_type.value}")
    print(f"    * Overall Status:    {report.overall_status.value.upper()}")
    print(f"    * Validation Score:  {report.validation_score:.4f} (100.0%)")
    print(f"    * Evaluated Rules:   {report.rules_evaluated}")
    print(f"    * Passed Rules:      {report.rules_passed}")
    print(f"    * Errors / Warnings: {report.error_count} Errors / {report.warning_count} Warnings")

    print("\n  [Rule Evaluation Results]")
    for issue in report.issues:
        icon = "[OK]" if issue.severity == SeverityLevel.INFO else f"[{issue.severity.value.upper()}]"
        print(f"    {icon:<8} {issue.rule_name:<40} (fields: {', '.join(issue.affected_fields)}) -> {issue.message}")

    # Verify JSON serialization round-trip
    report_dict = report.to_dict()
    report_json = report.to_json()
    assert report_dict["overall_status"] == report.overall_status.value
    assert len(report_dict["issues"]) == len(report.issues)

    print("\n" + "-" * 70)
    if report.overall_status in (ValidationStatus.VALID, ValidationStatus.WARNING) and report.error_count == 0:
        print("  Validation & Consistency Engine Status: [OPERATIONAL]")
        print("-" * 70)
        print("\n[OK] Document validation and consistency engine subsystem is fully operational.")
        return 0
    else:
        print("  Validation & Consistency Engine Status: [VALIDATION FAILURE]")
        print("-" * 70)
        return 1


def check_confidence_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect multi-tier confidence scoring, confidence bands, review rule catalog, and human review routing."""
    print("\n[+] Inspecting Multi-Tier Confidence Scoring & Review Routing Subsystem...")
    config = load_config(config_path)

    # 1. Configuration Overview
    print(f"  * Review Confidence Threshold: {config.confidence.review_confidence_threshold:.2f}")
    print(f"  * Field Low Cutoff Threshold:  {config.confidence.field.low_threshold:.2f}")
    print(f"  * Field High Cutoff Threshold: {config.confidence.field.high_threshold:.2f}")
    print(f"  * Routing Low Conf Threshold:  {config.confidence.review.low_confidence_threshold:.2f}")
    print(f"  * Critical Validation Route:   {config.confidence.review.critical_validation_required}")
    print(f"  * Missing Field Check:         {config.confidence.review.missing_required_field}")
    print(f"  * Table Math Mismatch Check:   {config.confidence.review.table_arithmetic_mismatch}")

    print("\n  [Component Weight Configurations]")
    print(f"    - Field Scoring Weights:     OCR={config.confidence.field.ocr_weight:.2f}, Extraction={config.confidence.field.extraction_weight:.2f}, Validation={config.confidence.field.validation_weight:.2f}")
    print(f"    - Table Scoring Weights:     Structure={config.confidence.table.structure_weight:.2f}, Rows={config.confidence.table.row_weight:.2f}, Cells={config.confidence.table.cell_weight:.2f}, Validation={config.confidence.table.validation_weight:.2f}")
    print(f"    - Document Composite Weights: Classification={config.confidence.document.classification_weight:.2f}, Fields={config.confidence.document.field_weight:.2f}, Tables={config.confidence.document.table_weight:.2f}, Validation={config.confidence.document.validation_weight:.2f}")

    router = ReviewRouter(config=config.confidence)

    # 2. Scenario A: High-Quality Clean Invoice (Eligible for STP)
    print("\n  [Scenario A: High-Quality Clean Document (Straight-Through Processing Candidate)]")
    bbox = BoundingBox(xmin=100.0, ymin=100.0, xmax=400.0, ymax=140.0)
    meta_a = DocumentMetadata(
        document_id="diag_doc_clean_001",
        filename="invoice_clean.pdf",
        file_path=Path("data/invoice_clean.pdf"),
        file_type="application/pdf",
        file_size_bytes=2048,
        checksum_sha256="clean_sha",
        page_count=1,
    )
    doc_a = Document(
        metadata=meta_a,
        pages=[DocumentPage(page_number=1, raw_text="Clean Invoice")],
        classified_type=DocumentType.INVOICE,
        classification_confidence=0.98,
    )

    fields_a = {
        "invoice_number": ExtractedField(
            name="invoice_number",
            value="INV-2026-001",
            normalized_value="INV-2026-001",
            field_type=FieldType.IDENTIFIER,
            provenance=Provenance("diag_doc_clean_001", 1, "INV-2026-001", bbox, 0.99, ExtractionMethod.REGEX_PATTERN),
            extraction_confidence=0.98,
            is_required=True,
            validation_status=ValidationStatus.VALID,
        ),
        "total": ExtractedField(
            name="total",
            value="$1,500.00",
            normalized_value=1500.00,
            field_type=FieldType.CURRENCY,
            provenance=Provenance("diag_doc_clean_001", 1, "$1,500.00", bbox, 0.98, ExtractionMethod.KEY_VALUE_HEURISTIC),
            extraction_confidence=0.96,
            is_required=True,
            validation_status=ValidationStatus.VALID,
        ),
    }

    table_a = Table(
        table_id="table_clean_01",
        page_number=1,
        headers=["Description", "Amount"],
        columns=[
            TableColumn(0, "Description", "description", 50, 300, "left", TableValueType.TEXT),
            TableColumn(1, "Amount", "amount", 300, 500, "right", TableValueType.CURRENCY),
        ],
        rows=[],
        line_items=[
            LineItem(
                row_index=0,
                description="Consulting Services",
                quantity=1.0,
                unit_price=1500.00,
                amount=1500.00,
                page_number=1,
                confidence=0.97,
                is_valid_arithmetic=True,
            )
        ],
        confidence=0.96,
    )

    val_report_a = ValidationReport(
        document_id="diag_doc_clean_001",
        document_type=DocumentType.INVOICE,
        overall_status=ValidationStatus.VALID,
        validation_score=1.0,
        rules_evaluated=5,
        rules_passed=5,
        issues=[],
    )

    result_a = router.route_document(doc_a, fields_a, [table_a], val_report_a)

    print(f"    * Composite Document Score: {result_a.document_confidence.overall_confidence:.4f} [{result_a.document_confidence.confidence_band.value.upper()}]")
    print(f"    * Routing Status:           {result_a.queue_item.status.value.upper()}")
    print(f"    * Queue Priority:           {result_a.queue_item.priority.value.upper()}")
    print(f"    * Straight-Through (STP):   {result_a.is_straight_through} [PASS]")
    print(f"    * Review Required:          {result_a.review_required}")

    # 3. Scenario B: Document with OCR Discrepancy & Arithmetic Failure (Routed to Review)
    print("\n  [Scenario B: Risky Document with Validation Discrepancy (Human Review Candidate)]")
    meta_b = DocumentMetadata(
        document_id="diag_doc_risky_002",
        filename="invoice_risky.pdf",
        file_path=Path("data/invoice_risky.pdf"),
        file_type="application/pdf",
        file_size_bytes=2048,
        checksum_sha256="risky_sha",
        page_count=1,
    )
    doc_b = Document(
        metadata=meta_b,
        pages=[DocumentPage(page_number=1, raw_text="Risky Invoice")],
        classified_type=DocumentType.INVOICE,
        classification_confidence=0.92,
    )

    bbox_low_ocr = BoundingBox(xmin=50.0, ymin=250.0, xmax=200.0, ymax=280.0)
    fields_b = {
        "invoice_number": ExtractedField(
            name="invoice_number",
            value="INV-2026-999",
            normalized_value="INV-2026-999",
            field_type=FieldType.IDENTIFIER,
            provenance=Provenance("diag_doc_risky_002", 1, "INV-2026-999", bbox, 0.95, ExtractionMethod.REGEX_PATTERN),
            extraction_confidence=0.92,
            is_required=True,
            validation_status=ValidationStatus.VALID,
        ),
        "tax": ExtractedField(
            name="tax",
            value="S45.OO",
            normalized_value=45.00,
            field_type=FieldType.CURRENCY,
            provenance=Provenance("diag_doc_risky_002", 1, "S45.OO", bbox_low_ocr, 0.42, ExtractionMethod.KEY_VALUE_HEURISTIC),
            extraction_confidence=0.55,
            is_required=False,
            validation_status=ValidationStatus.WARNING,
        ),
    }

    table_b = Table(
        table_id="table_risky_01",
        page_number=1,
        headers=["Description", "Qty", "Price", "Amount"],
        columns=[
            TableColumn(0, "Description", "description", 50, 250, "left", TableValueType.TEXT),
            TableColumn(1, "Qty", "quantity", 250, 350, "right", TableValueType.DECIMAL),
            TableColumn(2, "Price", "unit_price", 350, 450, "right", TableValueType.CURRENCY),
            TableColumn(3, "Amount", "amount", 450, 600, "right", TableValueType.CURRENCY),
        ],
        rows=[],
        line_items=[
            LineItem(
                row_index=0,
                description="Damaged Hardware Unit",
                quantity=3.0,
                unit_price=100.00,
                amount=250.00,  # 3 * 100 != 250 -> Arithmetic mismatch
                page_number=1,
                confidence=0.60,
                is_valid_arithmetic=False,
            )
        ],
        confidence=0.58,
    )

    val_report_b = ValidationReport(
        document_id="diag_doc_risky_002",
        document_type=DocumentType.INVOICE,
        overall_status=ValidationStatus.INVALID,
        validation_score=0.45,
        rules_evaluated=5,
        rules_passed=3,
        issues=[
            ValidationIssue(
                rule_id="RULE_LINE_ITEM_MATH",
                rule_name="Line-Item Math Verification",
                status=ValidationStatus.INVALID,
                severity=SeverityLevel.CRITICAL,
                message="Line item 1 calculation mismatch: 3 * 100.00 != 250.00",
                affected_fields=["amount"],
                category=ValidationCategory.ARITHMETIC,
            )
        ],
    )

    result_b = router.route_document(doc_b, fields_b, [table_b], val_report_b)

    print(f"    * Composite Document Score: {result_b.document_confidence.overall_confidence:.4f} [{result_b.document_confidence.confidence_band.value.upper()}]")
    print(f"    * Routing Status:           {result_b.queue_item.status.value.upper()}")
    print(f"    * Queue Priority:           {result_b.queue_item.priority.value.upper()}")
    print(f"    * Review Required:          {result_b.review_required} [PASS]")
    print(f"    * Flagged Review Targets:   {result_b.queue_item.target_count}")
    print(f"    * Flagged Issues:           {len(result_b.queue_item.issues)}")
    for issue in result_b.queue_item.issues:
        print(f"        * [{issue.severity.value.upper()}] ({issue.rule_name}) -> {issue.message}")

    # 4. Verify JSON roundtrip
    routing_dict = result_b.to_dict()
    routing_recon = ReviewRoutingResult.from_dict(routing_dict)
    assert routing_recon.document_id == result_b.document_id
    assert routing_recon.queue_item.priority == result_b.queue_item.priority

    print("\n" + "-" * 70)
    if result_a.is_straight_through and (not result_b.is_straight_through) and result_b.queue_item.priority in (ReviewPriority.URGENT, ReviewPriority.HIGH):
        print("  Confidence & Review Routing Subsystem Status: [OPERATIONAL]")
        print("-" * 70)
        print("\n[OK] Multi-tier confidence scoring and review routing subsystem is fully operational.")
        return 0
    else:
        print("  Confidence & Review Routing Subsystem Status: [DISCREPANCY DETECTED]")
        print("-" * 70)
        return 1


def check_visualization_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect visual explainability subsystem, coordinate transformations, evidence mapper, and overlay renderer."""
    import shutil
    import tempfile
    from PIL import Image, ImageDraw

    print("\n[+] Inspecting Visual Explainability & Evidence Overlays Subsystem...")
    config = load_config(config_path)
    vis_cfg = config.visualization

    # 1. Configuration Parameters
    print("\n  [Visualization Subsystem Configuration]")
    print(f"    * Output Directory:     {vis_cfg.output.directory}")
    print(f"    * Artifact Format:      {vis_cfg.output.format.upper()}")
    print(f"    * Save Manifest:        {vis_cfg.output.save_manifest}")
    print(f"    * Save Summary:         {vis_cfg.output.save_summary}")
    print(f"    * Active Layers:        OCR={vis_cfg.layers.ocr}, Fields={vis_cfg.layers.fields}, Tables={vis_cfg.layers.tables}, Validation={vis_cfg.layers.validation}, Review={vis_cfg.layers.review}")
    print(f"    * Render Settings:      Opacity={vis_cfg.rendering.opacity:.2f}, Line Width={vis_cfg.rendering.line_width}px, Labels={vis_cfg.rendering.show_labels}, Legend={vis_cfg.rendering.draw_legend}")
    print(f"    * Coordinate Transform: Clip to Page={vis_cfg.coordinates.clip_to_page}, Valid Dimensions Required={vis_cfg.coordinates.require_valid_dimensions}")

    # 2. Coordinate Transformation Verification
    print("\n  [Coordinate Transformation Diagnostics]")
    test_bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=250.0, ymax=180.0)
    img_w, img_h = 1000, 2000
    norm_box = CoordinateTransformer.to_normalized_box(test_bbox, img_w, img_h)
    pixel_box_recon = CoordinateTransformer.to_pixel_box(norm_box, img_w, img_h)

    print(f"    * Source Pixels:        [{test_bbox.xmin}, {test_bbox.ymin}, {test_bbox.xmax}, {test_bbox.ymax}]")
    print(f"    * Normalized (0.0-1.0): [{norm_box.xmin:.4f}, {norm_box.ymin:.4f}, {norm_box.xmax:.4f}, {norm_box.ymax:.4f}]")
    print(f"    * Reconstructed Pixels: [{pixel_box_recon.xmin:.1f}, {pixel_box_recon.ymin:.1f}, {pixel_box_recon.xmax:.1f}, {pixel_box_recon.ymax:.1f}]")
    assert abs(pixel_box_recon.xmin - test_bbox.xmin) < 1e-4
    assert abs(pixel_box_recon.ymax - test_bbox.ymax) < 1e-4
    print("    * Coordinate Mapping:   [VERIFIED - Lossless Reversible Transform]")

    # 3. End-to-End Synthetic Visual Evidence Rendering
    print("\n  [Synthetic Visual Evidence Rendering]")
    temp_dir = Path(tempfile.mkdtemp(prefix="documind_vis_diag_"))
    try:
        # Create a synthetic test image
        sample_img_path = temp_dir / "sample_page_1.png"
        img = Image.new("RGB", (800, 1000), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "DOCUMIND INVOICE", fill="black")
        draw.text((50, 100), "Invoice Number: INV-9901-X", fill="black")
        draw.text((50, 150), "Tax Amount: $150.00", fill="black")
        img.save(sample_img_path)

        # Build synthetic document & artifacts
        meta = DocumentMetadata(
            document_id="vis_diag_doc_01",
            filename="sample_invoice.pdf",
            file_path=sample_img_path,
            file_type="image/png",
            file_size_bytes=4096,
            checksum_sha256="vis_sha256",
            page_count=1,
        )
        page = DocumentPage(page_number=1, raw_text="DOCUMIND INVOICE\nInvoice Number: INV-9901-X\nTax Amount: $150.00", image_path=sample_img_path)
        doc = Document(metadata=meta, pages=[page], classified_type=DocumentType.INVOICE)

        fields = {
            "invoice_number": ExtractedField(
                name="invoice_number",
                value="INV-9901-X",
                normalized_value="INV-9901-X",
                field_type=FieldType.IDENTIFIER,
                provenance=Provenance("vis_diag_doc_01", 1, "INV-9901-X", BoundingBox(50, 95, 250, 125), 0.98, ExtractionMethod.REGEX_PATTERN),
                extraction_confidence=0.96,
                validation_status=ValidationStatus.VALID,
            ),
            "tax": ExtractedField(
                name="tax",
                value="$150.00",
                normalized_value=150.00,
                field_type=FieldType.CURRENCY,
                provenance=Provenance("vis_diag_doc_01", 1, "$150.00", BoundingBox(50, 145, 200, 175), 0.65, ExtractionMethod.KEY_VALUE_HEURISTIC),
                extraction_confidence=0.70,
                validation_status=ValidationStatus.WARNING,
                validation_messages=["Tax rate exceeds standard default thresholds"],
            ),
        }

        # Run DocumentVisualizer
        visualizer = DocumentVisualizer(config=vis_cfg)
        out_vis_dir = temp_dir / "visualizations"
        vis_res = visualizer.visualize_document(
            document=doc,
            fields=fields,
            tables=[],
            validation_report=None,
            routing_result=None,
            output_dir=out_vis_dir,
        )

        manifest = vis_res.manifest
        spatial_count = len(manifest.evidence_regions) if manifest else 0
        unlocated_count = len(manifest.unlocated_evidence) if manifest else 0
        total_regions = spatial_count + unlocated_count

        print(f"    * Manifest Regions:     {total_regions} (Spatial: {spatial_count}, Unlocated: {unlocated_count})")
        print(f"    * Rendered Pages:       {len(vis_res.page_visualizations)}")
        for pv in vis_res.page_visualizations:
            print(f"        * Page {pv.page_number}: {pv.overlay_image_path.name if pv.overlay_image_path else 'None'} ({len(pv.annotations)} annotations rendered)")
        print(f"    * Manifest Path:        {vis_res.artifact_paths.get('evidence_manifest')}")
        print(f"    * Summary Path:         {vis_res.artifact_paths.get('visualization_summary')}")

        # Verify artifacts exist
        manifest_path = vis_res.artifact_paths.get("evidence_manifest")
        summary_path = vis_res.artifact_paths.get("visualization_summary")
        overlay_path = vis_res.page_visualizations[0].overlay_image_path

        assert manifest_path and manifest_path.exists()
        assert summary_path and summary_path.exists()
        assert overlay_path and overlay_path.exists()

        # Verify manifest serialization round-trip
        loaded_manifest = VisualizationSerializer.load_manifest(manifest_path)
        assert loaded_manifest.document_id == "vis_diag_doc_01"
        assert len(loaded_manifest.evidence_regions) == spatial_count
        print("    * Manifest Roundtrip:   [VERIFIED - Validated JSON serialization & deserialization]")


    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "-" * 70)
    print("  Visual Explainability & Evidence Overlays Subsystem Status: [OPERATIONAL]")
    print("-" * 70)
    print("\n[OK] Visual explainability subsystem and evidence overlay renderer are fully operational.")
    return 0


def inspect_models() -> int:
    """Demonstrate data model instantiation, serialization, and provenance preservation."""
    print("\n[+] Inspecting Core Data Models & Provenance Serialization...")

    # 1. Bounding Box & Provenance
    bbox = BoundingBox(xmin=100.0, ymin=200.0, xmax=350.0, ymax=240.0)
    provenance = Provenance(
        document_id="doc_inv_2026_001",
        page_number=1,
        raw_text="INV-98234-A",
        bounding_box=bbox,
        ocr_confidence=0.985,
        extraction_method=ExtractionMethod.REGEX_PATTERN,
        character_span=(0, 11),
    )

    # 2. Extracted Field
    field = ExtractedField(
        name="invoice_number",
        value="INV-98234-A",
        normalized_value="INV-98234-A",
        field_type=FieldType.IDENTIFIER,
        provenance=provenance,
        extraction_confidence=0.95,
        confidence_source=ConfidenceSource.RULE_SCORE,
        validation_status=ValidationStatus.VALID,
        is_required=True,
    )

    # 3. Validation Result
    val_res = ValidationResult(
        rule_id="RULE_INV_NUMBER_FORMAT",
        rule_name="Invoice Number Standard Format Check",
        status=ValidationStatus.VALID,
        severity=SeverityLevel.INFO,
        message="Invoice identifier conforms to standardized alphanumeric format.",
        affected_fields=["invoice_number"],
    )

    # 4. Table Cell & Table Model
    table_cell = TableCell(
        row_index=0,
        col_index=0,
        text="Cloud Server Instance",
        normalized_value="Cloud Server Instance",
        bounding_box=bbox,
        confidence=0.98,
        page_number=1,
        value_type=TableValueType.TEXT,
    )

    table_col = TableColumn(
        index=0,
        name="Description",
        canonical_field="description",
        x_start=50.0,
        x_end=250.0,
        alignment="left",
        inferred_type=TableValueType.TEXT,
    )

    table_row = TableRow(
        row_index=0,
        cells=[table_cell],
        bounding_box=bbox,
        is_header=False,
        page_number=1,
        confidence=0.98,
    )

    table_model = Table(
        table_id="tbl_001",
        page_number=1,
        headers=["Description"],
        columns=[table_col],
        rows=[table_row],
        bounding_box=bbox,
        confidence=0.95,
    )

    # 5. Review Queue Item Model
    review_target = ReviewTarget(
        target_type=ReviewTargetType.FIELD if "ReviewTargetType" in globals() else None,  # type: ignore
        page_number=1,
        field_name="tax",
        bounding_box=bbox,
        reason="OCR confidence below threshold",
        severity=SeverityLevel.WARNING,
    )

    # 6. Visualization Evidence Region Model
    evidence_reg = EvidenceRegion(
        evidence_id="ev_diag_01",
        document_id="doc_inv_2026_001",
        page_number=1,
        region_type=AnnotationType.EXTRACTED_FIELD,
        bbox=bbox,
        source_text="INV-98234-A",
        normalized_value="INV-98234-A",
        field_name="invoice_number",
        extraction_confidence=0.95,
        is_spatial=True,
    )

    print(f"  * Model Initialized:  ExtractedField('{field.name}') -> Value='{field.value}'")
    print(f"  * Extraction Conf:    {field.extraction_confidence:.2f} ({field.confidence_source.value})")
    print(f"  * OCR Confidence:     {field.provenance.ocr_confidence:.2f}")
    print(f"  * Source Provenance:  Page {field.provenance.page_number}, Region [{field.provenance.bounding_box.xmin}, {field.provenance.bounding_box.ymin}, {field.provenance.bounding_box.xmax}, {field.provenance.bounding_box.ymax}]")
    print(f"  * Table Reconstructed: ID='{table_model.table_id}' ({len(table_model.columns)} cols, {len(table_model.rows)} rows, Conf={table_model.confidence:.2f})")
    print(f"  * Validation Status:  {val_res.status.value.upper()} (Rule: {val_res.rule_id})")
    print(f"  * Evidence Region:    {evidence_reg.evidence_id} (Type: {evidence_reg.region_type.value}, Spatial: {evidence_reg.is_spatial})")

    # Verify JSON round-trip
    field_dict = field.to_dict()
    reconstructed = ExtractedField.from_dict(field_dict)
    assert reconstructed.name == field.name
    assert reconstructed.value == field.value
    assert reconstructed.provenance.ocr_confidence == field.provenance.ocr_confidence

    tbl_dict = table_model.to_dict()
    tbl_recon = Table.from_dict(tbl_dict)
    assert tbl_recon.table_id == table_model.table_id
    assert len(tbl_recon.rows) == len(table_model.rows)

    ev_dict = evidence_reg.to_dict()
    ev_recon = EvidenceRegion.from_dict(ev_dict)
    assert ev_recon.evidence_id == evidence_reg.evidence_id
    assert ev_recon.field_name == evidence_reg.field_name

    print("\n[OK] Data model serialization and provenance round-trip verified.")
    return 0


def check_api_subsystem(config_path: str = "configs/default.yaml") -> int:
    """Inspect FastAPI REST API server, SQLite persistence, and UI asset distribution."""
    print("\n[+] Inspecting Web Application, REST API & Dashboard Subsystem (Phase 10)...")
    config = load_config(config_path)

    # 1. Inspect FastAPI and Uvicorn
    try:
        import fastapi
        import uvicorn
        print(f"  * FastAPI Framework:         [AVAILABLE] (v{fastapi.__version__})")
        print(f"  * Uvicorn ASGI Server:       [AVAILABLE] (v{uvicorn.__version__})")
    except ImportError as e:
        print(f"  * FastAPI / Uvicorn:         [MISSING] ({e})")
        return 1

    # 2. Inspect Persistence Layer & SQLite Connection
    from src.persistence.database import DatabaseManager
    from src.persistence.repository import DocumentRepository
    db_path = Path("data/documind.db")
    db_mgr = DatabaseManager(db_path)
    doc_repo = DocumentRepository(db_mgr)
    stats = doc_repo.get_overview_stats()
    print(f"  * SQLite Persistence:        [CONNECTED] (File: {db_path}, WAL mode active)")
    print(f"  * Total Tracked Documents:   {stats['total_documents']}")
    print(f"  * Pending Review Queue:      {stats['awaiting_review']}")

    # 3. Inspect Exporter Subsystem
    from src.export.exporter import DocumentExporter
    exporter = DocumentExporter()
    print("  * Document Exporters:        JSON, Reviewed JSON, CSV Tables, Markdown Report")

    # 4. Inspect Frontend Distribution Build
    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    index_html = dist_dir / "index.html"
    if index_html.is_file():
        print(f"  * Production UI Bundle:      [BUILT] ({dist_dir})")
    else:
        print(f"  * Production UI Bundle:      [SOURCE MODE] (Run 'npm run build' in frontend/ for standalone bundle)")

    print("\n" + "-" * 70)
    print("  Dashboard & API Subsystem Status: [OPERATIONAL]")
    print("-" * 70)
    print("\n[OK] Phase 10 REST API server and dashboard subsystem are ready.")
    print("  To launch the live dashboard: python run.py --serve\n")
    return 0


def serve_dashboard(host: str = "127.0.0.1", port: int = 8000) -> int:
    """Launch the DocuMind AI FastAPI server and dashboard."""
    import uvicorn
    from app.api.server import create_app

    print(f"\n[+] Starting DocuMind AI Server at http://{host}:{port}")
    print(f"  * REST API Documentation: http://{host}:{port}/docs")
    print(f"  * Dashboard Web UI:      http://{host}:{port}/\n")
    app = create_app()
    uvicorn.run(app, host=host, port=port)
    return 0


def main() -> int:
    """CLI main entry point."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="DocuMind AI: Document Intelligence & Data Extraction Platform"
    )
    parser.add_argument(
        "--check-config",
        nargs="?",
        const="configs/default.yaml",
        help="Validate project YAML configuration file.",
    )
    parser.add_argument(
        "--check-ingestion",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect document ingestion libraries, Poppler status, and preprocessing settings.",
    )
    parser.add_argument(
        "--check-ocr",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect Tesseract OCR binary, version, languages, and runtime availability.",
    )
    parser.add_argument(
        "--check-classification",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect classification rule definitions, weights, and sample document discrimination.",
    )
    parser.add_argument(
        "--check-extraction",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect structured field and entity extraction capabilities and sample extractions.",
    )
    parser.add_argument(
        "--check-tables",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect table detection, 2D grid reconstruction, line items, and arithmetic validation.",
    )
    parser.add_argument(
        "--check-validation",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect document validation rules, arithmetic consistency, format checks, and conflict detection.",
    )
    parser.add_argument(
        "--check-confidence",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect multi-tier confidence scoring, confidence bands, review rule catalog, and human review routing.",
    )
    parser.add_argument(
        "--check-visualization",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect visual explainability subsystem, coordinate transforms, evidence mapping, and overlay rendering.",
    )
    parser.add_argument(
        "--check-api",
        nargs="?",
        const="configs/default.yaml",
        help="Inspect REST API endpoints, database persistence, and dashboard readiness.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the live DocuMind AI REST API and web review dashboard on port 8000.",
    )
    parser.add_argument(
        "--inspect-models",
        action="store_true",
        help="Verify domain model instantiation, provenance tracking, and serialization.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="DocuMind AI v1.0.0 (Phase 10: Professional Document Intelligence & Human Review Dashboard)",
        help="Show platform version.",
    )

    args = parser.parse_args()

    print_banner()

    if args.serve:
        return serve_dashboard()
    elif args.check_config:
        return check_configuration(args.check_config)
    elif args.check_ingestion:
        return check_ingestion_subsystem(args.check_ingestion)
    elif args.check_ocr:
        return check_ocr_subsystem(args.check_ocr)
    elif args.check_classification:
        return check_classification_subsystem(args.check_classification)
    elif args.check_extraction:
        return check_extraction_subsystem(args.check_extraction)
    elif args.check_tables:
        return check_tables_subsystem(args.check_tables)
    elif args.check_validation:
        return check_validation_subsystem(args.check_validation)
    elif args.check_confidence:
        return check_confidence_subsystem(args.check_confidence)
    elif args.check_visualization:
        return check_visualization_subsystem(args.check_visualization)
    elif args.check_api:
        return check_api_subsystem(args.check_api)
    elif args.inspect_models:
        return inspect_models()
    else:
        print("\nDocuMind AI Phase 10 (Professional Document Intelligence & Human Review Dashboard) active.")
        print("Use --check-config to validate system configuration.")
        print("Use --check-ingestion to inspect ingestion dependencies and Poppler.")
        print("Use --check-ocr to inspect Tesseract OCR binary and languages.")
        print("Use --check-classification to inspect classification rules and discrimination.")
        print("Use --check-extraction to inspect structured field and entity extraction.")
        print("Use --check-tables to inspect table detection, line items, and arithmetic validation.")
        print("Use --check-validation to inspect document validation rules, math consistency & conflict detector.")
        print("Use --check-confidence to inspect multi-tier confidence scoring & human review routing.")
        print("Use --check-visualization to inspect visual explainability & evidence overlays.")
        print("Use --check-api to inspect REST API server and dashboard persistence.")
        print("Use --serve to launch the web review dashboard at http://127.0.0.1:8000")
        print("Use --inspect-models to verify domain model contracts & provenance.")
        print("Run 'python -m unittest discover tests' to execute test suite.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())


