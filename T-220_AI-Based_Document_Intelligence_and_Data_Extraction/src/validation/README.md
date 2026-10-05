# DocuMind AI — Document Validation & Consistency Engine

## 1. Subsystem Overview

The **Document Validation & Consistency Engine** is a deterministic, explainable, and multi-layered verification subsystem designed to audit extracted document fields, entities, tables, and line items. It operates strictly on extracted structured data produced by upstream subsystems (Phase 5 Structured Extraction & Phase 6 Table Intelligence) without re-running OCR or modifying extracted raw values.

```
+----------------------------------------------------------------------------------------------------+
|                                    DocumentValidationEngine                                        |
+----------------------------------------------------------------------------------------------------+
       |                                                                                       |
       v                                                                                       v
+-----------------------------+                                                  +-----------------------------+
|    RuleRegistry / Suite     |                                                  |   ValidationReport Model    |
| - RequiredFieldValidator    |                                                  | - Overall Status            |
| - Format Validators         |      Evaluates against Fields, Tables,           | - Validation Score (0.0-1.0)|
| - Cross-Field Chronology    | -----------------------------------------------> | - Rules Evaluated/Passed    |
| - Arithmetic Reconcilers    |             and Document Context                 | - Issues List (Provenance)  |
| - Conflict Detectors        |                                                  | - In-place Field Statuses   |
| - Table Integrity Bridge    |                                                  +-----------------------------+
+-----------------------------+
```

---

## 2. Core Architectural Principles

1. **Deterministic Verification**: All arithmetic checks, date order comparisons, and format regular expressions use exact numerical bounds with configurable tolerances ($\pm \$0.05$ subtotal/tax, $\pm \$0.02$ line items).
2. **Explainable Scoring**: The validation score is computed via a transparent penalty-subtraction model:
   $$\text{Validation Score} = \max\left(0.0, \frac{\text{Passed Rules}}{\text{Evaluated Rules}} - \sum \text{Severity Penalties}\right)$$
   This score is strictly decoupled from character-level OCR confidence and pattern-matching extraction confidence.
3. **Strict Provenance Preservation**: Every `ValidationIssue` preserves source page number, bounding box, raw/normalized values, OCR confidence, extraction method, and diagnostic metadata.
4. **Non-Destructive In-Place Auditing**: Field validation statuses (`VALID`, `WARNING`, `INVALID`) and explanation messages are updated on `ExtractedField` instances without altering underlying data.
5. **Multi-Document Type Specialization**: Tailored validation suites for Invoices, Receipts, Forms, and General Documents.

---

## 3. Subsystem Components & Rule Catalog

### 3.1 Base Classes & Interfaces (`src/validation/base.py`)
- `BaseValidationRule`: Abstract contract defining `rule_id`, `rule_name`, `category`, `target_fields`, and `evaluate()`.
- `BaseFieldValidator`: Specialized rule targeting individual key-value fields.
- `BaseDocumentValidator`: Multi-field composite document validator.
- `BaseValidationEngine`: High-level engine contract.

### 3.2 Domain Models (`src/validation/models.py`)
- `ValidationCategory`: Categorization enum (`REQUIRED_FIELD`, `FORMAT`, `TYPE`, `CROSS_FIELD`, `ARITHMETIC`, `TABLE`, `CONFLICT`, `CONSISTENCY`, `BUSINESS_RULE`).
- `ValidationIssue`: Subclass of `src.core.models.ValidationResult` enriched with bounding boxes, page numbers, and provenance.
- `ValidationReport`: Comprehensive document assessment containing overall status, validation score, rules evaluated/passed counts, issue list, and field statuses map.

### 3.3 Validation Rule Implementations
| Rule ID | Name | Category | Target Fields | Description |
| :--- | :--- | :--- | :--- | :--- |
| `VAL_REQ_001` | Required Field Presence | `REQUIRED_FIELD` | Mandatory Doc Fields | Asserts presence and non-empty values of required fields per document type. |
| `VAL_FMT_001` | Comprehensive Format Validation | `FORMAT` | `*` | Validates dates (ISO/DMY/MDY, valid leap years, sane year range 1900–2100), currency amounts, emails, phone numbers, and alphanumeric identifiers. |
| `VAL_XFLD_DATE_001` | Invoice vs Due Date Chronology | `CROSS_FIELD` | `invoice_date`, `due_date` | Asserts that `invoice_date <= due_date`. |
| `VAL_XFLD_SUBTOTAL_TOTAL_001` | Subtotal vs Total Magnitude | `CROSS_FIELD` | `subtotal`, `total` | Asserts that non-negative `subtotal <= total`. |
| `VAL_XFLD_DOB_001` | Date of Birth Sanity | `CROSS_FIELD` | `date_of_birth`, `dob` | Asserts birth dates are in the past and within reasonable human lifespan (0–125 years). |
| `VAL_ARITH_SUBTOTAL_TAX_001` | Subtotal + Tax Grand Total | `ARITHMETIC` | `subtotal`, `tax`, `total`, `discount` | Validates: $\text{subtotal} + \text{tax} + \text{shipping} + \text{tip} - \text{discount} == \text{total}$. |
| `VAL_ARITH_LINE_ITEM_001` | Line Item Arithmetic | `ARITHMETIC` | `line_items` | Validates: $\text{quantity} \times \text{unit\_price} - \text{discount} == \text{amount}$ for every line item. |
| `VAL_ARITH_LINES_SUBTOTAL_001` | Line Items Sum vs Subtotal | `ARITHMETIC` | `subtotal`, `line_items` | Asserts that $\sum \text{line\_items.amount} == \text{subtotal}$. |
| `VAL_CONF_CANDIDATES_001` | Candidate Conflict Detection | `CONFLICT` | `*` | Detects competing candidate values with differing normalized values for the same field. |
| `VAL_TBL_INTEGRITY_001` | Table Structural & Math Integrity | `TABLE` | `table`, `line_items` | Bridges Phase 6 `TableValidationResult` into document-level validation report. |

---

## 4. CLI Diagnostic Verification

Execute the validation subsystem verification suite from the terminal:

```bash
python run.py --check-validation
```

Output includes:
- Active configuration parameters & arithmetic tolerances
- Complete rule catalog grouped by category
- Live execution on synthetic multi-field invoice with line items
- Rule evaluation results with severity badges (`[OK]`, `[WARNING]`, `[ERROR]`)
- Data model serialization and round-trip verification

---

## 5. Python API Usage Example

```python
from src.core.config import ValidationConfig
from src.core.types import DocumentType
from src.validation import DocumentValidationEngine

# 1. Initialize engine with configuration
config = ValidationConfig(
    strict_mode=False,
    subtotal_tax_tolerance=0.05,
    line_item_amount_tolerance=0.02,
)
engine = DocumentValidationEngine(config=config)

# 2. Run validation on extraction result and tables
report = engine.validate_extraction_result(
    extraction_result=extraction_result,
    table_result=table_result,
)

# 3. Inspect validation outcomes
print(f"Status: {report.overall_status.value}")
print(f"Validation Score: {report.validation_score:.2f}")
print(f"Errors: {report.error_count}, Warnings: {report.warning_count}")

for issue in report.issues:
    print(f"[{issue.severity.value}] {issue.rule_name}: {issue.message}")

# 4. Serialize to JSON / Dictionary
report_dict = report.to_dict()
report_json = report.to_json()
```
