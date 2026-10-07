# DocuMind AI — Document Classification Subsystem

The **Document Classification Subsystem** provides deterministic, explainable, and provenance-aware categorization of ingested multi-page documents based on optical character recognition (OCR) signals, layout heuristics, and structured lexical rules.

---

## 1. Architectural Overview

The classification pipeline follows an open, modular design that strictly separates optical recognition certainty from document category scoring:

```
                               Document Ingestion & OCR
                                          │
                                          ▼
                                   Document Object
                             (DocumentPage, OCRTextRegion)
                                          │
                                          ▼
                             BaseDocumentClassifier (ABC)
                                          │
                                          ▼
                            RuleBasedDocumentClassifier
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
            Lexical Signals         Regex Patterns         Negative Signals
         ("tax invoice", 4.0)   (r"\bINV-\d+\b", 3.0)     ("cashier", -2.5)
                  │                       │                       │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                            ClassificationEvidence List
                       (page_number, bbox, ocr_confidence)
                                          │
                                          ▼
                               Score Aggregation &
                            Ambiguity Margin Analysis
                                          │
                                          ▼
                               ClassificationResult
                    (document_type, confidence, candidates,
                      evidence, is_ambiguous, candidate_scores)
```

---

## 2. Supported Document Types

| Document Type | Target Enum | Key Signal Indicators | Negative Penalties |
| :--- | :--- | :--- | :--- |
| **Invoice** | `DocumentType.INVOICE` | `tax invoice`, `bill to`, `remit to`, `invoice date`, `due date`, `NET 30`, `INV-\d+` | `cashier`, `register #`, `change due`, `application form` |
| **Receipt** | `DocumentType.RECEIPT` | `store receipt`, `cashier`, `register #`, `terminal #`, `change due`, `cash tendered`, `thank you for shopping` | `bill to`, `remit to`, `NET 30`, `purchase order` |
| **Form** | `DocumentType.FORM` | `application form`, `registration form`, `please print`, `applicant name`, `DOB:`, `SSN:`, `signature of applicant` | `tax invoice`, `invoice number`, `cashier` |
| **General Document** | `DocumentType.GENERAL_DOCUMENT` | `memorandum`, `executive summary`, `table of contents`, `introduction`, `conclusion`, `annual report` | `tax invoice`, `cashier`, `application form` |
| **Unknown** | `DocumentType.UNKNOWN` | Assigned when score < `min_score` (2.0), evidence < `min_evidence_count`, or ambiguous margin | N/A |

---

## 3. Mathematical Scoring & Confidence Methodology

### 3.1 Raw Score Aggregation
For each candidate document type $c \in \{\text{INVOICE}, \text{RECEIPT}, \text{FORM}, \text{GENERAL\_DOCUMENT}\}$, the raw score $S_c$ is calculated across all pages $P$:

$$S_c = \max\left(0.0, \sum_{p \in P} \sum_{e \in E_{c, p}} w(e) \cdot \mu_{\text{header}}(e, p)\right)$$

Where:
- $w(e)$ is the predefined weight of rule $e$ (positive for confirming signals, negative for class-conflicting penalties).
- $\mu_{\text{header}}(e, p) = 1.25$ if $p = 1$ and match is located within the top 30% of text, else $1.0$.

### 3.2 Monotonic Bounded Confidence Formulation
The winner class is selected as $C_1 = \arg\max_c S_c$ with score $S_1$. If $S_1 \ge \text{min\_score}$ and evidence count $\ge \text{min\_evidence\_count}$:

$$\text{Confidence}(C_1) = \min\left(1.0, \frac{S_1}{S_1 + 2.5}\right)$$

This formula produces well-calibrated confidence values:
- $S_1 = 3.75 \implies \text{Confidence} \approx 0.60$
- $S_1 = 10.0 \implies \text{Confidence} \approx 0.80$
- $S_1 = 22.5 \implies \text{Confidence} \approx 0.90$

### 3.3 Ambiguity & Margin Evaluation
If the runner-up candidate $C_2$ has $S_2 \ge \text{min\_score}$, the relative margin is evaluated:

$$\text{Margin}(C_1, C_2) = \frac{S_1 - S_2}{S_1 + S_2}$$

If $\text{Margin}(C_1, C_2) < \text{ambiguity\_margin}$ (default $0.10$), the result is flagged with `is_ambiguous = True` and detailed `ambiguity_reason`. If confidence falls below `confidence_threshold`, the document safely defaults to `UNKNOWN`.

---

## 4. Provenance Preservation

Every `ClassificationEvidence` retains strict source provenance linking back to physical page elements:
- `page_number`: 1-indexed source page.
- `matched_text`: Exact character sequence recognized.
- `character_span`: `(start_idx, end_idx)` in page text.
- `bounding_box`: Pixel-level coordinates of detected tokens (`xmin, ymin, xmax, ymax`).
- `ocr_confidence`: Mean OCR recognition confidence of the matching tokens.

```python
from src.classification import RuleBasedDocumentClassifier
from src.core.models import Document

classifier = RuleBasedDocumentClassifier()
result = classifier.classify(document)

print(f"Class: {result.document_type.value} (Conf: {result.confidence:.2f})")
for ev in result.get_top_evidence(limit=3):
    print(f"  Signal: {ev.signal_name} (Weight: {ev.weight}, Box: {ev.bounding_box})")
```

---

## 5. Configuration Reference

```yaml
classification:
  default_type: unknown
  confidence_threshold: 0.60      # Minimum confidence required to accept class
  min_score: 2.0                 # Minimum positive evidence score
  min_evidence_count: 1          # Minimum distinct evidence signals
  ambiguity_margin: 0.10         # Margin below which competing classes trigger ambiguity
  header_weight_multiplier: 1.25 # Multiplier for signals in top 30% of page 1
  keyword_weights_path: null     # Optional path to external custom rules YAML
```
