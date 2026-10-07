export type DocumentStatus = 
  | 'uploaded' 
  | 'processing' 
  | 'completed' 
  | 'needs_review' 
  | 'failed'
  | 'UPLOADED'
  | 'PROCESSING'
  | 'PROCESSED'
  | 'PENDING_REVIEW'
  | 'APPROVED'
  | 'REJECTED'
  | 'FAILED';

export type DocumentType = 
  | 'invoice' 
  | 'receipt' 
  | 'form' 
  | 'general_document' 
  | 'unknown'
  | 'INVOICE'
  | 'RECEIPT'
  | 'FORM'
  | 'GENERAL_DOCUMENT'
  | 'UNKNOWN';

export type ConfidenceBand = 'high' | 'medium' | 'low' | 'very_low' | 'HIGH' | 'MEDIUM' | 'LOW' | 'VERY_LOW';

export type ValidationSeverity = 'info' | 'warning' | 'error' | 'critical' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export type ValidationStatus = 'valid' | 'warning' | 'invalid' | 'unknown' | 'VALID' | 'WARNING' | 'INVALID' | 'UNKNOWN';

export type ReviewPriority = 'urgent' | 'high' | 'medium' | 'low' | 'none' | 'URGENT' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE' | 'CRITICAL';

export interface BoundingBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

export interface Provenance {
  document_id?: string;
  page_number?: number;
  raw_text?: string;
  bounding_box?: BoundingBox;
  ocr_confidence?: number;
  extraction_method?: string;
  character_span?: [number, number];
}

export interface ExtractedField {
  name: string;
  value: any;
  normalized_value?: any;
  field_type?: string;
  provenance?: Provenance;
  extraction_confidence?: number;
  confidence_source?: string;
  validation_status?: ValidationStatus;
  validation_messages?: string[];
  is_required?: boolean;
  is_human_corrected?: boolean;
  original_extracted_value?: any;
  correction_reason?: string;
  corrected_by?: string;
}

export type ExtractedFieldItem = ExtractedField;

export interface ExtractedEntity {
  entity_type: string;
  raw_text: string;
  normalized_value: any;
  confidence: number;
  page_number: number;
  bounding_box?: BoundingBox;
  char_span?: [number, number];
}

export interface TableCell {
  row_index: number;
  col_index: number;
  text: string;
  normalized_value?: any;
  bounding_box?: BoundingBox;
  confidence: number;
  page_number: number;
  value_type?: string;
}

export interface TableLineItem {
  row_index: number;
  page_number: number;
  item_code?: string;
  description: string;
  quantity?: number;
  unit_price?: number;
  discount?: number;
  tax?: number;
  amount?: number;
  confidence: number;
  is_valid_arithmetic: boolean;
}

export interface TableColumn {
  index: number;
  name: string;
  canonical_field?: string;
  x_start: number;
  x_end: number;
  alignment: string;
  inferred_type: string;
}

export interface Table {
  table_id: string;
  page_number: number;
  headers: string[];
  columns: TableColumn[];
  rows: { cells: TableCell[]; row_index: number; is_header: boolean; bounding_box?: BoundingBox; confidence: number }[];
  line_items: TableLineItem[];
  bounding_box?: BoundingBox;
  confidence: number;
}

export interface ValidationIssue {
  rule_id: string;
  rule_name: string;
  status: ValidationStatus;
  severity: ValidationSeverity;
  message: string;
  affected_fields: string[];
  category?: string;
  page_number?: number;
  bounding_box?: BoundingBox;
  raw_text?: string;
  suggested_fix?: string;
}

export interface ValidationReport {
  document_id: string;
  document_type: string;
  overall_status: ValidationStatus;
  validation_score: number;
  rules_evaluated: number;
  rules_passed: number;
  issues: ValidationIssue[];
}

export interface FieldConfidence {
  field_name: string;
  overall_confidence: number;
  confidence_band: ConfidenceBand;
  ocr_confidence: number;
  extraction_confidence: number;
  validation_score: number;
  review_required: boolean;
}

export interface DocumentConfidence {
  document_id?: string;
  overall_confidence?: number;
  composite_score?: number;
  confidence_band?: ConfidenceBand;
  classification_confidence?: number;
  mean_field_confidence?: number;
  mean_table_confidence?: number;
  validation_confidence?: number;
  review_required?: boolean;
  signals?: Record<string, number>;
  confidence_band_counts?: Record<string, number>;
  field_confidences?: Record<string, FieldConfidence>;
}

export interface ReviewTarget {
  target_type: string;
  page_number: number;
  field_name?: string;
  table_id?: string;
  row_index?: number;
  column_name?: string;
  bounding_box?: BoundingBox;
  reason?: string;
  severity?: ValidationSeverity;
}

export interface ReviewQueueItem {
  document_id: string;
  status: string;
  priority: ReviewPriority;
  targets: ReviewTarget[];
  target_count: number;
  issues: ValidationIssue[];
  created_at: string;
  review_reasons: string[];
  is_straight_through: boolean;
}

export interface EvidenceRegion {
  evidence_id: string;
  document_id: string;
  page_number: number;
  region_type: string;
  bbox?: BoundingBox;
  source_text?: string;
  normalized_value?: any;
  field_name?: string;
  table_id?: string;
  row_index?: number;
  column_name?: string;
  ocr_confidence?: number;
  extraction_confidence?: number;
  aggregated_confidence?: number;
  confidence_band?: ConfidenceBand;
  validation_status?: ValidationStatus;
  review_required?: boolean;
  review_reasons?: string[];
  is_spatial: boolean;
}

export interface OverlayAnnotation {
  annotation_id: string;
  page_number: number;
  annotation_type: string;
  bounding_box?: BoundingBox;
  label?: string;
  tooltip?: string;
  severity?: ValidationSeverity;
  confidence_band?: ConfidenceBand;
  confidence_score?: number;
  field_name?: string;
  table_id?: string;
  row_index?: number;
  column_name?: string;
  review_reasons?: string[];
  color_rgba: [number, number, number, number];
  fill_rgba?: [number, number, number, number];
  line_width: number;
  badge_text?: string;
  is_visible: boolean;
}

export interface DocumentRecord {
  id: string;
  document_id?: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  page_count: number;
  uploaded_at?: string;
  created_at?: string;
  status: DocumentStatus;
  document_type?: DocumentType;
  confidence?: number;
  confidence_score?: number;
  confidence_band?: ConfidenceBand;
  validation_status?: ValidationStatus;
  review_priority?: ReviewPriority;
  review_status?: string;
  target_count?: number;
  error_message?: string;
  updated_at?: string;
  pages?: { page_number: number; image_filename: string; has_overlay: boolean }[];
}

export type DocumentItem = DocumentRecord;

export interface DocumentDetail extends DocumentRecord {
  extraction_results?: {
    fields: ExtractedField[];
    entities?: ExtractedEntity[];
  };
  table_results?: {
    tables: Table[];
  };
  validation_results?: ValidationReport;
  confidence_results?: DocumentConfidence;
  visualizations?: {
    overlay_annotations: OverlayAnnotation[];
    evidence_regions: EvidenceRegion[];
  };
  audit_trail?: AuditRecord[];
  corrections?: HumanCorrection[];
  decisions?: ReviewDecision[];
  ocr_text?: string;
}

export interface HumanCorrection {
  correction_id: string;
  document_id: string;
  field_name: string;
  original_value?: string;
  corrected_value: string;
  reviewer_id: string;
  reason?: string;
  created_at: string;
  is_applied: boolean;
}

export interface FieldCorrectionPayload {
  field_name: string;
  original_value?: string;
  corrected_value: string;
  reviewer_id?: string;
  reason?: string;
}

export interface ReviewDecision {
  decision_id: string;
  document_id: string;
  action: string;
  reviewer_id: string;
  notes?: string;
  created_at: string;
  updated_document_status: string;
  updated_review_status: string;
}

export interface ReviewDecisionPayload {
  action: 'confirm' | 'override' | 'approve' | 'reject' | string;
  reviewer_id?: string;
  notes?: string;
}

export interface AuditRecord {
  audit_id: string;
  document_id: string;
  event_type: string;
  description: string;
  actor: string;
  details_json?: string;
  timestamp: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  phase: string;
  ocr_ready: boolean;
  ocr_engine: string;
  tesseract_version?: string;
  poppler_detected: boolean;
  database_ready: boolean;
  storage_directories_ready: boolean;
  ocr_status: {
    is_ready: boolean;
    tesseract_version?: string;
    poppler_available?: boolean;
  };
  timestamp?: string;
}

export type SystemHealth = HealthStatus;

export interface OverviewStats {
  total_documents: number;
  completed_documents: number;
  processed_documents?: number;
  pending_review_documents?: number;
  approved_documents?: number;
  awaiting_review: number;
  failed_documents: number;
  validation_errors_count: number;
  straight_through_processing_rate_pct?: number;
  average_confidence_pct?: number;
  priority_breakdown: Record<string, number>;
  document_type_distribution: Record<string, number>;
  by_type?: Record<string, number>;
  by_priority?: Record<string, number>;
  recent_documents: DocumentRecord[];
}

export type SystemStats = OverviewStats;
