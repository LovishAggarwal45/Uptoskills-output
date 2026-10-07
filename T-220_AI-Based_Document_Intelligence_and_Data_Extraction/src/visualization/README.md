# DocuMind AI — Visual Explainability & Evidence Overlays Subsystem

## Overview

The **Visual Explainability & Evidence Overlays Subsystem** (`src/visualization/`) delivers end-to-end visual explainability, spatial traceability, and review routing visualization for DocuMind AI. It bridges extracted values, OCR words, tabular structures, arithmetic validations, and confidence/routing decisions with the underlying physical document layout.

---

## Key Architectural Principles

1. **Zero Fabrication**:
   - The subsystem **never** invents bounding boxes, coordinates, or spatial regions.
   - Missing coordinates or unlocated validation findings are preserved in `manifest.unlocated_evidence` rather than drawing deceptive bounding boxes.
2. **Lossless Coordinate Transformations**:
   - `CoordinateTransformer` provides strict, bidirectional conversion between normalized unit coordinates $[0.0, 1.0]$ and absolute pixel coordinates $(X, Y)$.
   - Safely handles image scaling, DPI adjustments, and out-of-bounds boundary clipping.
3. **Multi-Layer Explainability**:
   - Granular visual layers can be independently toggled: OCR words, key-value fields, named entities, table grids/cells, validation findings, and human review targets.
4. **Accessible Visual Encoding**:
   - Visual labels incorporate textual badges (`[FLD]`, `[TBL]`, `[REVIEW]`, `[URGENT]`, `[MATH-ERR]`) in addition to distinct semantic colors, ensuring accessibility for color-blind reviewers and grayscale rendering.
5. **Non-Destructive Rendering**:
   - Original document images in `data/` and `data/processed/` are never overwritten. Overlays are saved as dedicated artifact images (`page_00X_overlay.png`).
6. **Machine-Readable Evidence Manifest**:
   - Full provenance, confidence scores, validation statuses, and review rationales are exported to `evidence_manifest.json` and summarized in `visualization_summary.json`.

---

## Subsystem Architecture

```
src/visualization/
├── __init__.py                # Public API exports
├── base.py                    # Subsystem interfaces (BaseVisualizer, DocumentVisualizer coordinator)
├── coordinates.py             # Geometric coordinate transformations & clipping
├── evidence_mapper.py         # Multi-phase evidence extraction & compilation
├── exceptions.py              # Custom visualization exceptions hierarchy
├── field_overlay.py           # Key-value field and named entity annotation builder
├── legend.py                  # Color palettes, accessible badge symbols, & legend banner renderer
├── models.py                  # Dataclasses & Enums (EvidenceRegion, OverlayAnnotation, Manifest, Result)
├── overlay_renderer.py        # Alpha blending, outline drawing, badge pill rendering, & artifact export
├── review_overlay.py          # Human review target and validation alert annotation builder
├── serializer.py              # JSON serialization & deserialization for manifests and summaries
└── table_overlay.py           # Table grid, cell, and line-item arithmetic mismatch annotation builder
```

---

## Semantic Color Palette & Badge Symbols

| Annotation Category | Color (RGBA) | Badge Symbol | Purpose |
|---------------------|--------------|--------------|---------|
| `OCR_WORD` | `(180, 180, 180, 120)` | `[OCR]` | Raw word-level OCR token bounding box |
| `EXTRACTED_FIELD` | `(33, 150, 243, 220)` | `[FLD]` | High/Medium confidence structured field |
| `EXTRACTED_ENTITY` | `(156, 39, 176, 220)` | `[ENT]` | Generic named entity (Date, Org, Amount) |
| `TABLE_BOUNDS` | `(0, 150, 136, 200)` | `[TBL]` | Outer table perimeter bounding box |
| `TABLE_CELL` | `(77, 182, 172, 160)` | `[CEL]` | Individual reconstructed table cell |
| `TABLE_LINE_ITEM` (Valid) | `(46, 125, 50, 200)` | `[ITEM]` | Verified tabular line item |
| `TABLE_LINE_ITEM` (Math Error) | `(211, 47, 47, 240)` | `[MATH-ERR]` | Line item with arithmetic discrepancy $(Qty \times Price \neq Amount)$ |
| `VALIDATION_WARNING` | `(255, 152, 0, 220)` | `[WARN]` | Soft validation warning or format deviation |
| `VALIDATION_ERROR` | `(244, 67, 54, 240)` | `[ERR]` | Rule violation or checksum mismatch |
| `REVIEW_TARGET` | `(255, 112, 67, 230)` | `[REVIEW]` | Low confidence or flagged human review candidate |
| `REVIEW_CRITICAL` | `(183, 28, 28, 255)` | `[URGENT]` | Critical escalation requiring immediate manual intervention |

---

## Programmatic Usage

```python
from src.core.config import load_config
from src.visualization import DocumentVisualizer

config = load_config("configs/default.yaml")
visualizer = DocumentVisualizer(config=config.visualization)

# Generate visual explainability artifacts
vis_result = visualizer.visualize_document(
    document=doc,
    fields=fields,
    tables=tables,
    validation_report=val_report,
    routing_result=routing_result,
    ocr_result=ocr_result,
    output_dir=Path("outputs/visualizations"),
)

print(f"Total Pages Rendered: {len(vis_result.page_visualizations)}")
print(f"Manifest Path: {vis_result.artifact_paths['evidence_manifest']}")
print(f"Overlay Path: {vis_result.page_visualizations[0].overlay_image_path}")
```

---

## CLI Diagnostic

Run the automated visual explainability diagnostic suite:

```bash
python run.py --check-visualization
```
