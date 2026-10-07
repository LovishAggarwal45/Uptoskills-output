# DocuMind AI — Multi-Tier Confidence Scoring & Human Review Routing

## 1. Subsystem Overview

The **Multi-Tier Confidence Scoring & Human Review Routing Subsystem** synthesizes and orchestrates the signals produced across upstream pipeline phases (Phase 3 OCR, Phase 4 Classification, Phase 5 Extraction, Phase 6 Table Intelligence, and Phase 7 Validation) into explainable, deterministic confidence scores and intelligent human-in-the-loop review routing.

```
+----------------------------------------------------------------------------------------------------+
|                                           ReviewRouter                                             |
+----------------------------------------------------------------------------------------------------+
       |                                       |                                       |
       v                                       v                                       v
+-----------------------------+ +-----------------------------+ +-----------------------------+
|  FieldConfidenceCalculator  | |  TableConfidenceCalculator  | | DocumentConfidenceCalculator|
| - OCR token confidence      | | - OCR cell confidence       | | - Classification score      |
| - Extraction score          | | - Geometry alignment        | | - Mean field confidence     |
| - Format validation status  | | - Line-item math validation | | - Mean table confidence     |
| - Candidate conflicts       | | - Table subtotal validation | | - Document validation score |
+-----------------------------+ +-----------------------------+ +-----------------------------+
       \                                       |                                       /
        \______________________________________v______________________________________/
                                               |
                                               v
+----------------------------------------------------------------------------------------------------+
|                                        ReviewRuleCatalog                                           |
| - LowFieldConfidenceRule      - MissingRequiredFieldRule      - UnclassifiedDocumentRule          |
| - LowOCRConfidenceRule        - CandidateConflictRule         - TableArithmeticRule               |
| - ValidationReportRule                                                                             |
+----------------------------------------------------------------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------------+
|                                        PriorityEvaluator                                           |
| - URGENT: Critical validation failure / arithmetic mismatch / high-risk invalid data               |
| - HIGH:   Missing required fields / low document confidence (<0.70) / classification errors        |
| - MEDIUM: Candidate conflicts / validation warnings / medium confidence (0.40 - 0.70)              |
| - LOW:    Straight-Through Processing (STP) eligible / high confidence (>= 0.85)                   |
+----------------------------------------------------------------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------------+
|                                      ReviewRoutingResult                                           |
| - queue_item: ReviewQueueItem (priority, status, targets with bounding boxes, issues, reasons)      |
| - is_straight_through: bool (True -> AUTO_APPROVED, False -> PENDING review)                       |
| - document_confidence: DocumentConfidence (overall_confidence, confidence_band, breakdown)          |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Principles

1. **Zero Fabrication**: Missing upstream signals (e.g., table confidence on a document without tables) remain explicitly `None` and are never replaced with artificial defaults like 0.5 or 0.8.
2. **Dynamic Weight Renormalization**: When an input signal is absent (`None`), the aggregator recalculates active weights proportionally so their sum remains exactly $1.0$:
   $$w_i' = \frac{w_i}{\sum_{j \in \text{active}} w_j}$$
3. **Decoupled Confidence & Validity Dimensions**: High optical/extraction confidence does not imply business correctness. A clearly recognized line item with invalid math ($2 \times \$100 = \$500$) maintains high OCR confidence ($0.98$) but triggers validation failure and critical human review routing.
4. **Validation-Aware Critical Overrides**: If document validation fails with `INVALID` status or `CRITICAL` severity issues, the system overrides high confidence and mandates review routing.
5. **Full Spatial Provenance**: Every `ReviewTarget` identifies the exact page number, field name or table/row coordinates, bounding box (`BoundingBox`), trigger reason, and severity level for UI overlay highlighting.

---

## 3. Confidence Bands & Formulas

Confidence scores across all levels are strictly bounded in $[0.0, 1.0]$ and categorized into four standardized confidence bands:

| Confidence Band | Score Range | Description | Typical Action |
| :--- | :--- | :--- | :--- |
| **`HIGH`** | $[0.85, 1.00]$ | Exceptionally clear text, verified format, valid math. | Candidate for Straight-Through Processing (STP). |
| **`MEDIUM`** | $[0.65, 0.85)$ | Acceptable extraction with minor warning or medium OCR. | Standard processing or low-priority sampling. |
| **`LOW`** | $[0.40, 0.65)$ | Below threshold, missing secondary signals, or minor conflict. | Standard human review queue. |
| **`VERY_LOW`** | $[0.00, 0.40)$ | Poor OCR quality, severe discrepancies, or unrecognized layout. | Urgent review queue. |

### 3.1 Field-Level Confidence Formula
$$\text{Field Confidence} = \max\left(0.0, \left(w_{\text{ocr}} \cdot s_{\text{ocr}} + w_{\text{ext}} \cdot s_{\text{ext}} + w_{\text{val}} \cdot s_{\text{val}}\right) - P_{\text{conflict}}\right)$$

### 3.2 Table & Line-Item Confidence Formula
$$\text{Table Confidence} = w_{\text{struct}} \cdot s_{\text{struct}} + w_{\text{row}} \cdot s_{\text{row}} + w_{\text{cell}} \cdot s_{\text{cell}} + w_{\text{val}} \cdot s_{\text{val}}$$

### 3.3 Document-Level Composite Confidence Formula
$$\text{Doc Confidence} = w_{\text{clf}} \cdot s_{\text{clf}} + w_{\text{fld}} \cdot \bar{s}_{\text{fld}} + w_{\text{tbl}} \cdot \bar{s}_{\text{tbl}} + w_{\text{val}} \cdot s_{\text{val}}$$

---

## 4. Review Routing & Priority Matrix

| Review Priority | Trigger Condition | Recommended Reviewer Action |
| :--- | :--- | :--- |
| **`URGENT`** | Critical validation issue, line item arithmetic mismatch, or subtotal calculation error. | Immediate intervention; block downstream financial posting. |
| **`HIGH`** | Missing required document field, document score $< 0.70$, or unclassified document type. | Priority manual entry and verification of key fields. |
| **`MEDIUM`** | Candidate value conflict, validation warning, or medium OCR confidence. | Standard visual spot-check of highlighted regions. |
| **`LOW`** | Clean document, high confidence ($\ge 0.85$), all validation checks passed. | Eligible for Straight-Through Processing (STP) / `AUTO_APPROVED`. |

---

## 5. Subsystem Components

### 5.1 Domain Models (`src/confidence/models.py`)
- `ConfidenceBand`: Enum (`HIGH`, `MEDIUM`, `LOW`, `VERY_LOW`).
- `ReviewPriority`: Enum (`LOW`, `MEDIUM`, `HIGH`, `URGENT`).
- `ReviewStatus`: Enum (`PENDING`, `IN_REVIEW`, `APPROVED`, `REJECTED`, `AUTO_APPROVED`).
- `ReviewTargetType`: Enum (`FIELD`, `TABLE`, `ROW`, `CELL`, `DOCUMENT`).
- `FieldConfidence`, `LineItemConfidence`, `TableConfidence`, `DocumentConfidence`: Granular confidence models with full component score breakdowns and explainable text.
- `ReviewTarget`: Spatial provenance target with `BoundingBox`, page number, and reason.
- `ReviewIssue`: Individual granular issue triggering review.
- `ReviewQueueItem`: Human-reviewer work item placed in the review queue.
- `ReviewRoutingResult`: Complete result aggregate returned by `ReviewRouter`.

### 5.2 Calculators & Aggregator
- `FieldConfidenceCalculator` (`src/confidence/field_confidence.py`): Calculates field scores with candidate conflict penalties.
- `TableConfidenceCalculator` (`src/confidence/table_confidence.py`): Calculates line item and table grid scores.
- `DocumentConfidenceCalculator` (`src/confidence/document_confidence.py`): Calculates composite document scores.
- `aggregate_weighted_signals` (`src/confidence/aggregator.py`): Dynamic weight renormalization engine.

### 5.3 Rule Catalog & Routing Engine
- `ReviewRuleCatalog` (`src/confidence/review_rules.py`): Declarative rule catalog evaluating conditions across all tiers.
- `PriorityEvaluator` (`src/confidence/priority.py`): Deterministic urgency classification.
- `ReviewRouter` (`src/confidence/review_router.py`): Master orchestrator dispatching `ReviewQueueItem` and STP decisions.

---

## 6. CLI Diagnostic Verification

Execute the confidence and review routing verification command from the terminal:

```bash
python run.py --check-confidence
```

Output includes:
- Active confidence configuration, thresholds, and component weights.
- Live execution on clean document showing Straight-Through Processing (`AUTO_APPROVED`, `LOW` priority).
- Live execution on risky document with OCR discrepancy and math error showing Human Review routing (`PENDING`, `URGENT` priority, 4 flagged targets with bounding boxes).
- JSON round-trip and serialization validation.

---

## 7. Python API Usage Example

```python
from src.core.config import ConfidenceConfig
from src.confidence import ReviewRouter

# 1. Initialize ReviewRouter with configuration
config = ConfidenceConfig(
    review_confidence_threshold=0.70,
    ocr_weight=0.35,
    extraction_weight=0.40,
    validation_weight=0.25,
)
router = ReviewRouter(config=config)

# 2. Evaluate document, extracted fields, tables, and validation report
result = router.route_document(
    document=document,
    fields=extracted_fields,
    tables=tables,
    validation_report=validation_report,
)

# 3. Inspect composite confidence and routing decisions
print(f"Overall Confidence: {result.document_confidence.overall_confidence:.4f}")
print(f"Confidence Band:    {result.document_confidence.confidence_band.value.upper()}")
print(f"Straight-Through:   {result.is_straight_through}")
print(f"Review Status:      {result.queue_item.status.value.upper()}")
print(f"Review Priority:    {result.queue_item.priority.value.upper()}")

# 4. Access review targets for visual bounding box highlighting
for target in result.queue_item.targets:
    print(f"Target on page {target.page_number} ({target.field_name}): {target.reason}")
    if target.bounding_box:
        print(f"  BBox: [{target.bounding_box.xmin}, {target.bounding_box.ymin}, {target.bounding_box.xmax}, {target.bounding_box.ymax}]")

# 5. Export JSON for review queue microservices or UI consumption
queue_item_json = result.queue_item.to_json()
```
