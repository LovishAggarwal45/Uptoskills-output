# DocuMind AI — Table Intelligence & Line-Item Extraction Subsystem (Phase 6)

## Overview

The `tables` subsystem provides a production-grade, deterministic table intelligence and line-item extraction engine for DocuMind AI. Operating on real OCR tokens, horizontal lines, and spatial layout geometry, it detects tabular regions, reconstructs 2D grid structures, maps columns to canonical business fields, extracts itemized records, performs mathematical cross-validation, and detects multi-page table continuation.

---

## Architectural Principles

1. **Spatial & Geometric Layout Reconstruction**:
   - Reconstructs tables via X/Y coordinate clustering, horizontal baseline alignment, and vertical column gap detection rather than brittle regular expressions.
2. **Deterministic Column & Header Mapping**:
   - Matches header tokens against an extensive dictionary of canonical accounting synonyms (`description`, `item_code`, `quantity`, `unit_price`, `amount`, `tax`, `discount`, `unit_of_measure`).
3. **Multi-Line Wrapped Description Handling**:
   - Resolves wrapped multi-line cell descriptions by analyzing vertical proximity and absence of numeric tokens on subsequent description lines.
4. **Summary Row Discrimination**:
   - Distinguishes data rows from summary footer rows (`Subtotal`, `Tax`, `Total`, `Balance Due`, `Notes`).
5. **Mathematical Integrity Validation**:
   - Validates row-level arithmetic (`quantity * unit_price - discount == amount`).
   - Validates table-level sums against reported subtotals and grand totals (`sum(amounts) == subtotal`, `subtotal + tax == total`).
6. **Multi-Page Table Continuation**:
   - Detects when tables span across page breaks using column geometry alignment and header similarity.
7. **Explainable Composite Confidence**:
   - Computes weighted confidence scores from header detection, column sharpness, row regularity, arithmetic consistency, and OCR confidence.
8. **Complete Source Provenance**:
   - Preserves 2D bounding boxes, page numbers, raw OCR tokens, and OCR confidence down to every table cell and line item.

---

## Subsystem Architecture

```
src/tables/
├── __init__.py                # Public subsystem API exports
├── base.py                    # Abstract base interfaces (BaseTableDetector, BaseTableReconstructor, etc.)
├── models.py                  # Strongly-typed domain models (Table, TableRow, TableCell, LineItem, etc.)
├── exceptions.py              # Domain exceptions (TableExtractionError, TableStructureError, etc.)
├── header_detector.py         # Header candidate detection and canonical synonym matching
├── column_parser.py           # Column boundary discovery, X-clustering, and type inference
├── row_parser.py              # Horizontal row parsing, multi-line wrapping, and summary detection
├── cell_parser.py             # Cell value typing, normalization, and confidence aggregation
├── structure.py               # 2D table grid assembly and coordinate validation
├── detector.py                # Table region detection and false positive suppression
├── line_item_extractor.py     # Canonical line-item extraction and semantic column mapping
├── validators.py              # Mathematical and table sum integrity validation
├── continuation.py            # Multi-page table continuation detection and linking
├── confidence.py              # Explainable composite table confidence scoring
└── extractor.py               # Master TableExtractor coordinator
```

---

## CLI Diagnostic Command

Run the table intelligence diagnostic command to verify table detection, grid reconstruction, line-item extraction, and arithmetic validation:

```bash
python run.py --check-tables
```
