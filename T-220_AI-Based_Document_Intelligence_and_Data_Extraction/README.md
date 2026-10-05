# DocuMind AI: Document Intelligence & Data Extraction Platform

**DocuMind AI** is an enterprise-grade, research-oriented document intelligence platform engineered in Python. It provides end-to-end processing for PDFs, high-resolution scans, and document images—combining modular preprocessing, optical character recognition (OCR), document structure understanding, structured entity/table extraction, deterministic rule validation, confidence scoring, provenance tracking, and human-in-the-loop review routing.

---

## 1. Architectural Overview & System Flow

```
[ Ingest Document ] (PDF / Multi-page TIFF / PNG / JPEG)
         │  ├── Format Sniffing & Binary Magic Bytes
         │  ├── SHA-256 Checksum Provenance
         │  └── Multi-Page Stage Extraction (page_001_original.png)
         ▼
[ Modular Preprocessing ] (OpenCV Pipeline)
         │  ├── Skew Angle Estimation (minAreaRect & Hough Lines)
         │  ├── Center Rotation with Canvas Expansion
         │  ├── Grayscale Normalization (Alpha-Channel Aware)
         │  ├── CLAHE Adaptive Contrast Enhancement
         │  ├── Non-Local Means / Bilateral Denoising
         │  └── Diagnostic Quality Heuristics (Laplacian Blur Variance)
         ▼
[ Pluggable OCR Subsystem ] (Tesseract Engine / BaseOCREngine)
         │  ├── Word-level Tokenization with Native Raw Confidence (0.0 - 100.0)
         │  ├── Normalized Optical Confidence Scores ([0.0, 1.0])
         │  ├── Reading-Order Line Clustering (Top-to-Bottom, Left-to-Right)
         │  ├── Structural Line & Block Bounding Box Unions (BoundingBox)
         │  └── Deterministic Layout-Preserving Text Reconstruction
         ▼
[ Document Classification ] (Invoice, Receipt, Form, General Document, Unknown)
         │
         ▼
[ Structured Extraction Engine ]
         ├── Key-Value Entities (Dates, Totals, Tax, Vendor, ID Numbers)
         └── Tabular Line Items (Headers, Rows, Columns, Bounding Coordinates)
         │
         ▼
[ Deterministic Validation Engine ]
         ├── Format Checks (ISO 8601 Dates, E.164 Phones, RFC 5322 Emails)
         ├── Required Field Completeness Checks
         └── Relational & Arithmetic Consistency Checks (Subtotal + Tax = Total)
         │
         ▼
[ Composite Confidence & Review Routing ]
         ├── OCR Quality Score (Raw Optical Confidence)
         ├── Extraction Confidence Score (Pattern/Heuristic Fit)
         └── Validation Status & Human-in-the-Loop Review Flags
         │
         ▼
[ Structured Export & Visualization ]
         ├── Standardized JSON (Full Schema + Provenance Metadata)
         ├── CSV Line Item Tables
         ├── Human-Readable Validation Diagnostic Reports
         └── Visual Provenance Overlays (Annotated BBoxes & Validation Badges)
```

---

## 2. Core Engineering Principles

1. **Production-Grade Modularity**: Clean separation of concerns across 10 specialized domain subsystems.
2. **Strict Provenance Preservation**: Every extracted value retains complete traceability back to its source:
   - Source document ID and 1-indexed page number
   - Original and processed page images (`page_001_original.png`, `page_001_processed.png`)
   - Word, line, and block Bounding Boxes (`BoundingBox`)
   - Native OCR confidence score (`ocr_confidence` and `raw_confidence`)
   - OCR engine name, engine version, and language metadata
3. **Tripartite Confidence Distinction**:
   - **OCR Confidence**: Physical optical quality and character recognition certainty.
   - **Extraction Confidence**: Semantic certainty that an extracted snippet matches the target field.
   - **Validation Status**: Formal logical/arithmetic correctness (`VALID`, `WARNING`, `INVALID`).
4. **No Destructive Mutation**: Original input files are never overwritten; intermediate page representations are versioned in `data/processed/{document_id}/`.
5. **No Fabricated Data or Metrics**: Does not simulate or fabricate artificial AI metrics. All OCR outputs reflect real recognition data or explicit documented diagnostics when dependencies are missing.
6. **Deterministic Rule Validation**: Arithmetic, format, and relational integrity are evaluated through deterministic rules.
7. **Hardware & Deployment Friendly**: Designed for standard CPU development environments without requiring paid cloud APIs.

---

## 3. Directory Structure

```
documind-ai/
├── app/                        # Application & Presentation Layer
│   ├── ui/                     # UI Views & Dashboards
│   └── components/             # Reusable Visual Components
├── src/                        # Core Engine & Domain Logic
│   ├── core/                   # Shared Models, Configuration, Logging, Exceptions
│   │   ├── models.py           # Typed Data Classes (Document, DocumentPage, ExtractedField, BoundingBox, etc.)
│   │   ├── types.py            # Enums (DocumentType, FieldType, ValidationStatus, OCRLevel, etc.)
│   │   ├── config.py           # Configuration Loaders & Schema Specifications
│   │   ├── logging.py          # Structured Contextual Logging System
│   │   └── exceptions.py       # Domain Exception Hierarchy
│   ├── ingestion/              # Document Ingestion & Page Extraction
│   │   ├── document_loader.py  # Unified Ingestion Coordinator (BaseIngestionEngine)
│   │   ├── image_ingestion.py  # PNG, JPG, TIFF, BMP Ingestion Handler
│   │   ├── pdf_ingestion.py    # Multi-page PDF Ingestion (pypdf + pdf2image)
│   │   └── validators.py       # Magic Byte Sniffing, Checksums & Validation
│   ├── preprocessing/          # CV Image Enhancement & Deskewing
│   │   ├── opencv_preprocessor.py # OpenCV Preprocessing Pipeline (BaseImagePreprocessor)
│   │   ├── operations.py       # Grayscale, Resizing, CLAHE, Denoise, Thresholding
│   │   ├── deskew.py           # MinAreaRect / Hough Line Skew Estimation & Canvas Expansion
│   │   └── quality.py          # Variance of Laplacian Blur & Quality Heuristics
│   ├── ocr/                    # Pluggable OCR Subsystem
│   │   ├── base.py             # BaseOCREngine abstract contract
│   │   ├── models.py           # Structured OCR Models (OCRWord, OCRLine, OCRBlock, PageOCRResult, DocumentOCRResult)
│   │   ├── tesseract_engine.py # Tesseract OCR Engine (pytesseract wrapper)
│   │   ├── engine_factory.py   # OCREngineFactory for dynamic backend instantiation
│   │   ├── processor.py        # DocumentOCRProcessor (Multi-page orchestration & DocumentPage updates)
│   │   ├── postprocessing.py   # Reading Order Sorting, Confidence Normalization & Text Reconstruction
│   │   ├── availability.py     # Rigorous Tesseract binary, version, and language detection
│   │   └── exceptions.py       # OCRException, OCREngineUnavailableError, OCRInvalidImageError
│   ├── classification/         # Document Classification Subsystem
│   │   ├── base.py             # BaseDocumentClassifier abstract contract
│   │   ├── models.py           # ClassificationResult, ClassificationEvidence, ClassificationCandidate, SignalType
│   │   ├── classifier.py       # RuleBasedDocumentClassifier (Deterministic scoring, ambiguity margin, provenance)
│   │   ├── rules.py            # 140+ structured rules for INVOICE, RECEIPT, FORM, GENERAL_DOCUMENT
│   │   ├── exceptions.py       # ClassificationError, InvalidClassificationInput, ClassificationConfigurationError
│   │   └── README.md           # Mathematical Scoring, Architecture & Configuration Manual
│   ├── extraction/             # Key-Value, Entity, and Structured Extractors
│   │   ├── base.py             # BaseDocumentExtractor, BaseFieldExtractor, BaseEntityExtractor contracts
│   │   ├── models.py           # ExtractedField, ExtractedEntity, ExtractionCandidate, ExtractionResult, EntityType
│   │   ├── patterns.py         # Precompiled Regexes (Dates, Money, IDs, Emails, Phones) & Label Dictionaries
│   │   ├── normalizers.py      # Normalization Layer (Amounts, ISO Dates, Phones, Emails, Clean Identifiers)
│   │   ├── spatial.py          # 2D Geometric Layout, Spatial Proximity & Bounding-Box Envelope Merging
│   │   ├── confidence.py       # 5-Factor Explainable Confidence Calculation Engine
│   │   ├── entity_extractors.py# GenericEntityExtractor (Dates, Money, Emails, Phones, IDs, Orgs, Addresses)
│   │   ├── invoice_extractor.py# InvoiceExtractor (Invoice #, Dates, Amounts, Tax ID, Terms, PO, Parties)
│   │   ├── receipt_extractor.py# ReceiptExtractor (Merchant, Receipt #, Amounts, Tax, Payment Method, Time)
│   │   ├── form_extractor.py   # FormExtractor (Applicant, Dates, Contact, Address, Form #, Signature)
│   │   ├── extractor.py        # DocumentExtractor Coordinator & Classification-Driven Dispatcher
│   │   └── README.md           # Comprehensive Subsystem Manual & Field Reference
│   ├── validation/             # Rule Engine & Deterministic Validators
│   ├── confidence/             # Composite Scoring & Review Flag Generation
│   ├── visualization/          # Annotation Rendering & Provenance Overlays
│   ├── export/                 # JSON, CSV, and Report Serializers
│   └── pipeline/               # Orchestration, Context Flow, and Runner
├── data/                       # Document Data Directories
│   ├── input/                  # Unprocessed Ingestion Queue
│   ├── samples/                # Reference & Sample Documents
│   └── processed/              # Processed Intermediate Files (data/processed/{doc_id}/)
├── outputs/                    # Export Destinations
│   ├── json/                   # Structured Extracted JSON Files
│   ├── csv/                    # Extracted Tabular CSV Data
│   ├── reports/                # Markdown / Text Validation Reports
│   └── visualizations/         # Annotated Document Images
├── configs/
│   └── default.yaml            # Master Configuration Specification
├── tests/                      Automated Unit & Integration Tests (317 passed, 1 skipped; verified with pytest)
│   ├── test_models.py          # Model Geometry, Provenance, Serialization Tests
│   ├── test_config.py          # YAML Parsing & Config Validation Tests
│   ├── test_pipeline_contracts.py # Pipeline Orchestration & Step Tests
│   ├── test_validation_engine.py  # Rule Validation & Consistency Tests
│   ├── test_file_validation.py # File Integrity, Magic Bytes & Size Limits Tests
│   ├── test_image_ingestion.py # Single-page Image Ingestion & Color Mode Tests
│   ├── test_pdf_ingestion.py   # Multi-page PDF Structure & Poppler Diagnostics
│   ├── test_preprocessing.py   # Grayscale, Resize, CLAHE, Denoise, Deskew Tests
│   ├── test_quality_metrics.py # Laplacian Variance & Diagnostic Warnings Tests
│   ├── test_ingestion_integration.py # Ingestion to Preprocessing Integration Test
│   ├── test_ocr_models.py      # OCRWord/Line/Block, Reading Order & Text Reconstruction Tests
│   ├── test_ocr_availability.py# Binary Discovery & Availability Diagnostics Tests
│   ├── test_ocr_engine.py      # Engine Factory, Parsing & Custom Engine Tests
│   ├── test_ocr_processor.py   # Multi-page Document OCR Orchestration Tests
│   ├── test_ocr_integration.py # Live/Conditional OCR Integration Test
│   ├── test_classification_models.py     # Evidence, Candidate & Result Model Tests
│   ├── test_classification_rules.py      # Lexical, Regex & Negative Signal Matching Tests
│   ├── test_classification_classifier.py # Multi-Class, Multi-Page, Ambiguity & Margin Tests
│   ├── test_classification_integration.py# Full Pipeline Ingestion -> OCR -> Classification Test
│   ├── test_extraction_models.py         # ExtractedField/Entity/Candidate/Result Serialization Tests
│   ├── test_extraction_patterns.py       # Regex Patterns & Label Dictionary Matching Tests
│   ├── test_extraction_normalizers.py    # Money, ISO Date, Phone, Email & ID Normalization Tests
│   ├── test_field_extractors.py          # Spatial Layout, BBox Envelopes & Proximity Pairing Tests
│   ├── test_entity_extractors.py         # Generic Entity Multi-Type Extraction & Provenance Tests
│   ├── test_invoice_extractor.py         # Commercial Invoice Complete Field & Multi-Page Tests
│   ├── test_receipt_extractor.py         # Retail POS Receipt Complete Field & Cashier Tests
│   ├── test_form_extractor.py            # Membership/Intake Form Field & Signature Tests
│   ├── test_extraction_classifier_integration.py # Classifier-Driven Extraction Dispatcher Tests
│   └── test_extraction_integration.py    # Live E2E Ingestion -> OCR -> Classification -> Extraction Test
├── requirements.txt            # Explicit, Justified Dependencies
├── README.md                   # Technical Documentation & Architecture Manual
└── run.py                      # CLI Diagnostic Utility & Entry Point
```

---

## 4. OCR Subsystem Architecture (Phase 3)

### OCR Engine Abstraction & Factory
- **Pluggable Base Contract ([`BaseOCREngine`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/base.py))**: Decouples document processing logic from specific OCR libraries.
- **Factory Pattern ([`OCREngineFactory`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/engine_factory.py))**: Enables switching or registering alternative backends (`tesseract`, `easyocr`, etc.).
- **Tesseract Backend ([`TesseractOCREngine`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/tesseract_engine.py))**: Executes OCR via `pytesseract.image_to_data`, extracting word bounding boxes, recognition confidences, block numbers, and line numbers.

### Structured Hierarchy & Reading Order
- **[`OCRWord`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/models.py#L22)**: Individual token preserving `raw_confidence` (0-100), `confidence` (0.0-1.0), and `BoundingBox(xmin, ymin, xmax, ymax)`.
- **[`OCRLine`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/models.py#L64)**: Line cluster aggregating child words, computing unified bounding boxes and mean word confidences.
- **[`OCRBlock`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/models.py#L104)**: Coherent paragraph block aggregating lines.
- **Reading Order ([`sort_reading_order`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/postprocessing.py#L48))**: Deterministic clustering sorting lines top-to-bottom and words left-to-right, assigning sequential reading indices without non-deterministic LLM reordering.
- **Text Reconstruction ([`reconstruct_page_text`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/ocr/postprocessing.py#L96))**: Formats text preserving line breaks and paragraph spacing.

---

## 5. Document Classification Subsystem (Phase 4)

- **Pluggable Base Contract ([`BaseDocumentClassifier`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/classification/base.py))**: Clean abstraction for document classification.
- **Rule-Based Engine ([`RuleBasedDocumentClassifier`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/classification/classifier.py))**:
  - Deterministic evaluation of 140+ structured rules across `INVOICE`, `RECEIPT`, `FORM`, `GENERAL_DOCUMENT`, and `UNKNOWN`.
  - Positive phrase/keyword matching, regex pattern identification, and negative penalty signals.
  - Page 1 header positioning boost ($\times 1.25$).
  - Monotonic bounded confidence formula: $\text{Confidence} = \frac{S}{S + 2.5}$.
  - Ambiguity margin detection: flags conflicting signals when score difference $< 10\%$.
- **Explainable Evidence & Spatial Provenance ([`ClassificationEvidence`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/classification/models.py))**:
  - Each matched signal captures rule weight, source page, character span, bounding box, and OCR confidence.

---

## 6. Structured Field & Entity Extraction Subsystem (Phase 5)

- **Classification-Driven Dispatcher ([`DocumentExtractor`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/extraction/extractor.py))**:
  - Routes documents to specialized extractors (`InvoiceExtractor`, `ReceiptExtractor`, `FormExtractor`) based on predicted `DocumentType`.
  - Safely falls back to `GenericEntityExtractor` on `GENERAL_DOCUMENT` and `UNKNOWN` without forcing non-existent billing fields.
- **Specialized Schema Extractors**:
  - **Invoices**: `invoice_number`, `invoice_date`, `due_date`, `vendor_name`, `customer_name`, `subtotal`, `tax`, `total`, `currency`, `payment_terms`, `purchase_order_number`, `tax_id`.
  - **Receipts**: `merchant_name`, `receipt_number`, `transaction_date`, `transaction_time`, `cashier`, `terminal_id`, `subtotal`, `tax`, `total`, `payment_method`, `currency`.
  - **Forms**: `applicant_name`, `first_name`, `last_name`, `date_of_birth`, `phone`, `email`, `address`, `form_number`, `signature_indicator`.
- **2D Spatial Layout Matching ([`src/extraction/spatial.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/extraction/spatial.py))**:
  - Inline colon separation, horizontal adjacency matching ($gap \le 350\text{px}$), and vertically-below value detection ($gap \le 60\text{px}$).
- **Data Normalization Layer ([`src/extraction/normalizers.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/extraction/normalizers.py))**:
  - Standardizes currency amounts (US/UK `$1,250.00`, European `1.250,50 €`), ISO-8601 dates with ambiguity detection, E.164 phone numbers, lowercase emails, and clean identifiers without altering `raw_value`.
- **5-Factor Explainable Extraction Confidence**:
  $$\text{Confidence} = 0.35 \cdot S_{\text{label}} + 0.30 \cdot S_{\text{pattern}} + 0.15 \cdot S_{\text{spatial}} + 0.10 \cdot S_{\text{uniqueness}} + 0.10 \cdot S_{\text{ocr}}$$

---

## 7. Table Intelligence & Line-Item Extraction Subsystem (Phase 6)

- **2D Spatial Layout & Grid Reconstruction ([`src/tables/structure.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/tables/structure.py))**:
  - Reconstructs tables via X/Y coordinate clustering, baseline proximity, and vertical whitespace gap detection without relying on fragile regex or fake OCR.
- **Canonical Header & Synonym Matching ([`src/tables/header_detector.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/tables/header_detector.py))**:
  - Matches column labels against an extensive dictionary of canonical accounting synonyms (`description`, `item_code`, `quantity`, `unit_price`, `amount`, `tax`, `discount`, `unit_of_measure`).
- **Wrapped Multi-Line Description Merging ([`src/tables/row_parser.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/tables/row_parser.py))**:
  - Robustly merges multi-line product descriptions into single logical table rows by verifying baseline proximity and absence of numeric tokens on continuation lines.
- **Summary Row Discrimination**:
  - Distinguishes data rows from summary footer rows (`Subtotal`, `Tax`, `Total`, `Balance Due`, `Notes`).
- **Deterministic Arithmetic Validation ([`src/tables/validators.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/tables/validators.py))**:
  - Validates row-level calculations ($\text{Quantity} \times \text{Unit Price} - \text{Discount} \approx \text{Amount}$).
  - Validates table sums ($\sum \text{Line Amounts} \approx \text{Subtotal}$ and $\text{Subtotal} + \text{Tax} \approx \text{Total}$).
- **Multi-Page Table Continuation Detection ([`src/tables/continuation.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/tables/continuation.py))**:
  - Identifies tables continuing across page breaks through column geometry overlap ($\text{IoU} \ge 0.65$) and header similarity.
- **Explainable Composite Table Confidence ([`src/tables/confidence.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/tables/confidence.py))**:
  - Provides clear signal score breakdowns across header recognition, column alignment, row regularity, arithmetic consistency, and OCR confidence.

---

## 8. Document Validation & Consistency Engine (Phase 7)

- **Deterministic Rule Engine ([`src/validation/validator.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/validator.py))**:
  - Main coordinator executing deterministic and heuristic verification across fields, tables, line items, and document metadata.
- **Required Field Verification ([`src/validation/required_field_validator.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/required_field_validator.py))**:
  - Enforces mandatory fields per document schema (`INVOICE`, `RECEIPT`, `FORM`, `GENERAL_DOCUMENT`, `UNKNOWN`).
- **Comprehensive Format Validation ([`src/validation/format_validators.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/format_validators.py))**:
  - Validates dates (leap years, ISO/DMY/MDY syntax, sane year boundaries), currency amounts, emails, phone numbers, and alphanumeric identifiers.
- **Cross-Field Relational & Chronological Consistency ([`src/validation/cross_field_validators.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/cross_field_validators.py))**:
  - Chronology: asserts $\text{invoice\_date} \le \text{due\_date}$.
  - Magnitude: asserts non-negative $\text{subtotal} \le \text{total}$.
  - Demographics: asserts $\text{date\_of\_birth}$ is in the past within realistic human lifespan ($0 \le \text{age} \le 125$).
- **Mathematical & Arithmetic Reconciliation ([`src/validation/arithmetic_validators.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/arithmetic_validators.py))**:
  - Reconciles totals: $\text{subtotal} + \text{tax} + \text{shipping} + \text{tip} - \text{discount} == \text{total}$.
  - Reconciles line items: $\text{quantity} \times \text{unit\_price} - \text{discount} == \text{amount}$.
  - Reconciles subtotal: $\sum \text{line\_items.amount} == \text{subtotal}$.
- **Candidate Conflict & Ambiguity Detection ([`src/validation/conflict_detector.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/conflict_detector.py))**:
  - Identifies distinct competing candidates with differing normalized values for the same field and flags review warnings.
- **Table Integrity Bridge ([`src/validation/table_validators.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/table_validators.py))**:
  - Bridges Phase 6 `TableValidationResult` into document-level `ValidationReport`.
- **Explainable Validation Scoring ([`src/validation/confidence.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/validation/confidence.py))**:
  - Transparent penalty subtraction model strictly distinct from OCR character confidence and extraction candidate confidence.

---

## 9. Multi-Tier Confidence Scoring & Human Review Routing Subsystem (Phase 8)

- **Multi-Tier Composite Architecture ([`src/confidence/review_router.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/review_router.py))**:
  - Orchestrates signals across OCR, classification, extraction, tables, and validation into unified confidence scores bounded in $[0.0, 1.0]$.
- **Zero-Fabrication Signal Aggregation ([`src/confidence/aggregator.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/aggregator.py))**:
  - Dynamically renormalizes weights when individual signals (e.g., table confidence or OCR confidence) are missing without fabricating dummy numbers.
- **Granular Calculators**:
  - **Field Confidence ([`src/confidence/field_confidence.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/field_confidence.py))**: Combines OCR tokens, extraction pattern match, format validation, and candidate conflict penalties.
  - **Table Confidence ([`src/confidence/table_confidence.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/table_confidence.py))**: Evaluates structure, row regularity, cell recognition, and arithmetic validation at line-item and table levels.
  - **Document Confidence ([`src/confidence/document_confidence.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/document_confidence.py))**: Synthesizes classification, field mean, table mean, and document validation score.
- **Standardized Confidence Bands**:
  - Categorizes all scores into `HIGH` ($\ge 0.85$), `MEDIUM` ($0.65 - 0.85$), `LOW` ($0.40 - 0.65$), and `VERY_LOW` ($< 0.40$).
- **Declarative Review Rule Catalog ([`src/confidence/review_rules.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/review_rules.py))**:
  - Comprehensive suite of rules triggering human review for low field/OCR confidence, missing required fields, candidate conflicts, arithmetic mismatches, validation failures, and unclassified documents.
- **Validation-Aware Review Routing & Urgency Priority ([`src/confidence/priority.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/confidence/priority.py))**:
  - Decouples confidence from validity. Validation errors and calculation mismatches override high OCR confidence and route to `URGENT` review.
  - Generates `ReviewQueueItem` with explicit spatial `ReviewTarget` coordinates (`BoundingBox`), issue reasons, and Straight-Through Processing (`AUTO_APPROVED`) decisions.

---

## 10. Visual Explainability & Evidence Overlays Subsystem (Phase 9)

- **Explainable Evidence Compilation ([`src/visualization/evidence_mapper.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/evidence_mapper.py))**:
  - Compiles full spatial and non-spatial evidence across OCR words, key-value fields, named entities, table cells, line-item calculations, validation issues, and human review targets.
- **Zero Fabrication & Lossless Coordinates ([`src/visualization/coordinates.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/coordinates.py))**:
  - Bidirectional coordinate mapping between normalized unit coordinates $[0.0, 1.0]$ and pixel dimensions $(X, Y)$ with strict page boundary clipping.
  - Never fabricates fake coordinates; missing or non-spatial evidence is cleanly registered in `manifest.unlocated_evidence`.
- **Specialized Overlay Builders**:
  - **Field Overlays ([`src/visualization/field_overlay.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/field_overlay.py))**: Renders bounding boxes and label badges with confidence scores for extracted fields and entities.
  - **Table Overlays ([`src/visualization/table_overlay.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/table_overlay.py))**: Visualizes outer table bounds, cell grids, and arithmetic mismatches (`[MATH-ERR]`).
  - **Review Overlays ([`src/visualization/review_overlay.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/review_overlay.py))**: Highlights human review targets and validation errors with high-visibility alerts (`[REVIEW]`, `[URGENT]`).
- **Accessible Multi-Layer Renderer ([`src/visualization/overlay_renderer.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/overlay_renderer.py))**:
  - Alpha blending, priority Z-sorting (review targets drawn above fields/OCR), pill badge labels, and dynamic legend banner attachment.
- **Evidence Manifest & Summary Serializer ([`src/visualization/serializer.py`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/visualization/serializer.py))**:
  - Exports machine-readable `evidence_manifest.json` and high-level `visualization_summary.json`.

---

## 11. Professional Document Intelligence & Human Review Dashboard (Phase 10)

- **Modern Web Application & Presentation Layer ([`frontend/`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/frontend))**:
  - Built with **React 18**, **TypeScript**, **Vite**, **Tailwind CSS**, and **Lucide React**.
  - Multi-view navigation: Live Stats Overview, Document Inventory, Human Review Queue, Split-Screen Workspace, and Subsystem Health Monitor.
  - Interactive split-screen review workspace featuring:
    - Side-by-side original page vs. Phase 9 visual evidence overlay toggle.
    - Synchronized bounding box highlighting across fields and tables on hover.
    - Confidence gauge visualization (`HIGH`, `MEDIUM`, `LOW`, `VERY_LOW`).
    - Structured field inspection and inline correction modal with reason capture.
    - Tabular line-item grid with cell-level confidence and arithmetic mismatch indicators.
    - Validation error and warning issue panel with affected field tags.
    - Reviewer action bar (`APPROVE`, `CONFIRM`, `OVERRIDE`, `REJECT`) with reviewer notes.
    - Chronological, immutable audit trail panel logging all system lifecycle and human reviewer events.
- **FastAPI High-Performance REST API ([`app/api/`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/app/api))**:
  - Production ASGI backend with interactive Swagger OpenAPI documentation at `/api/docs` and ReDoc at `/api/redoc`.
  - Static file mounting serving the compiled production frontend bundle at `/` with clean client-side SPA routing fallback.
- **SQLite Persistence & Audit Trail ([`src/persistence/`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/persistence))**:
  - Thread-safe connection pooling, Write-Ahead Logging (`PRAGMA journal_mode = WAL`), and foreign key enforcement.
  - Relational schema tracking `documents`, `human_corrections`, `review_decisions`, and `audit_trail`.
- **Multi-Format Export Subsystem ([`src/export/`](file:///C:/Users/Jnapikachowdary/.gemini/antigravity/scratch/documind-ai/src/export))**:
  - **Machine JSON**: Complete schema output with original extraction values, bounding boxes, and provenance.
  - **Reviewed JSON**: Includes original extracted values, human correction overrides, reviewer metadata, and decision history.
  - **Tabular CSV**: Exports extracted tables as standard CSV files, or an in-memory ZIP bundle for multi-table documents.
  - **Markdown Integrity Report**: Human-readable executive summary of validation issues, confidence breakdown, and review rationale.

### REST API Reference Table

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Subsystem health, Tesseract status, DB connectivity, and version |
| `GET` | `/api/stats/overview` | Live aggregated platform metrics and queue distribution |
| `GET` | `/api/documents` | List documents with pagination, status filters, and search queries |
| `POST` | `/api/documents` | Upload PDF or image with format/size validation and auto-processing option |
| `GET` | `/api/documents/{id}` | Retrieve complete document record and structured extraction result |
| `POST` | `/api/documents/{id}/process` | Trigger end-to-end processing pipeline on an uploaded document |
| `GET` | `/api/documents/{id}/pages/{page}/image` | Fetch original extracted page image |
| `GET` | `/api/documents/{id}/pages/{page}/overlay` | Fetch Phase 9 rendered visual evidence overlay image |
| `GET` | `/api/documents/{id}/evidence-manifest` | Fetch Phase 9 spatial evidence manifest and unlocated issues |
| `GET` | `/api/review-queue` | Query prioritized human review queue items with filter parameters |
| `GET` | `/api/documents/{id}/corrections` | List human field corrections applied to document |
| `POST` | `/api/documents/{id}/corrections` | Submit human field value override with reason |
| `POST` | `/api/documents/{id}/review-decision` | Record formal reviewer decision (`APPROVE`, `OVERRIDE`, `REJECT`, etc.) |
| `GET` | `/api/documents/{id}/audit-trail` | Fetch full chronological audit history for document |
| `GET` | `/api/documents/{id}/export/json` | Download original machine extraction JSON |
| `GET` | `/api/documents/{id}/export/reviewed` | Download reviewed JSON with applied overrides and audit trail |
| `GET` | `/api/documents/{id}/export/csv` | Download table CSV file or multi-table ZIP archive |
| `GET` | `/api/documents/{id}/export/report` | Download human-readable Markdown validation summary |

---

## 12. Getting Started & Verification

### Running the Live Web Application & Dashboard
```bash
# Launch the FastAPI server and web dashboard on http://localhost:8000
python run.py --serve
```

### Running the CLI Diagnostic Tools
```bash
# Verify configuration parsing and settings
python run.py --check-config

# Inspect document ingestion libraries, Poppler status, and preprocessing settings
python run.py --check-ingestion

# Inspect Tesseract OCR binary, version, languages, and runtime availability
python run.py --check-ocr

# Inspect classification rule definitions, weights, and sample document discrimination
python run.py --check-classification

# Inspect structured field and entity extraction across document schemas
python run.py --check-extraction

# Inspect table detection, 2D grid reconstruction, line items, and arithmetic validation
python run.py --check-tables

# Inspect document validation rules, arithmetic consistency, format checks, and conflict detection
python run.py --check-validation

# Inspect multi-tier confidence scoring, confidence bands, review rule catalog, and human review routing
python run.py --check-confidence

# Inspect visual explainability, coordinate transformations, evidence mapping, and overlay rendering
python run.py --check-visualization

# Inspect REST API, persistence layer, exporter subsystem, and web dashboard readiness
python run.py --check-api

# Inspect data models, geometry, and provenance serialization
python run.py --inspect-models
```

### Running the Complete Test Suite
```bash
python -m pytest tests -q
```

---

## 13. Implementation Roadmap

- [x] **Phase 1: Foundation & Architectural Contracts** (Typed Models, Config, Logging, Exceptions, Pipeline Contracts)
- [x] **Phase 2: Document Ingestion & Image Preprocessing** (Format Sniffing, Staging, Multi-Page PDFs, Deskew, CLAHE, Quality Diagnostics)
- [x] **Phase 3: Production-Quality OCR Subsystem** (Tesseract Engine, Word/Line/Block Regions, Bounding Boxes, Confidence Normalization, Reading Order, Multi-Page Processor)
- [x] **Phase 4: Document Classification Subsystem** (Rule-based and Keyword-Heuristic Classifier for Invoices, Receipts, Forms, General Documents, Unknown)
- [x] **Phase 5: Structured Field & Entity Extraction** (Invoices, Receipts, Forms, Generic Entities, Spatial Layout Matching, Normalization Layer, Provenance Tracking)
- [x] **Phase 6: Tabular Line-Item Extraction Subsystem** (Grid, Borderless, Header Alignment, Multi-Line Cell Merging, Arithmetic Validation, Multi-Page Continuation)
- [x] **Phase 7: Document Validation & Consistency Engine** (Deterministic Rules, Math Reconcilers, Format & Chronology Checks, Conflict Detection, Table Math Integrity, Explainable Scoring)
- [x] **Phase 8: Confidence Scoring & Review Routing** (Multi-Tier Scoring, Zero-Fabrication Dynamic Renormalization, Confidence Bands, Validation-Aware Review Routing, Queue Items, Spatial Targets, STP)
- [x] **Phase 9: Visual Explainability & Evidence Overlays** (Coordinate Transformations, Multi-Layer Overlays, Accessible Badges, Review/Validation Highlights, Evidence Manifest & Summary JSON)
- [x] **Phase 10: Professional Document Intelligence & Human Review Dashboard** (React 18 SPA, FastAPI REST API, SQLite Persistence, Review Queue & Overrides, Multi-Format Exporters, Live Audit Trail)


