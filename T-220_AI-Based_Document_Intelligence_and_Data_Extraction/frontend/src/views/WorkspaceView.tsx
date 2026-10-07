import React, { useState } from 'react';
import { 
  ArrowLeft, 
  Download, 
  Layers, 
  Table as TableIcon, 
  ShieldAlert, 
  BarChart3, 
  History,
  FileText,
  Play
} from 'lucide-react';
import { 
  DocumentDetail, 
  BoundingBox, 
  ExtractedFieldItem,
  ReviewDecisionPayload,
  FieldCorrectionPayload
} from '../types';
import { StatusBadge, ReviewPriorityBadge } from '../components/StatusBadge';
import { DocumentViewer } from '../components/DocumentViewer';
import { FieldExtractionPanel } from '../components/FieldExtractionPanel';
import { TableReviewPanel } from '../components/TableReviewPanel';
import { ValidationPanel } from '../components/ValidationPanel';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { AuditTrailPanel } from '../components/AuditTrailPanel';
import { ReviewDecisionBar } from '../components/ReviewDecisionBar';
import { CorrectionModal } from '../components/CorrectionModal';
import { ExportModal } from '../components/ExportModal';

interface WorkspaceViewProps {
  document: DocumentDetail;
  isLoading: boolean;
  onBack: () => void;
  onProcess: (docId: string) => void;
  onSaveCorrection: (docId: string, payload: FieldCorrectionPayload) => Promise<void>;
  onSubmitDecision: (docId: string, payload: ReviewDecisionPayload) => Promise<void>;
  onRefresh: () => void;
}

type WorkspaceTab = 'fields' | 'tables' | 'validation' | 'confidence' | 'audit';

export const WorkspaceView: React.FC<WorkspaceViewProps> = ({
  document,
  isLoading,
  onBack,
  onProcess,
  onSaveCorrection,
  onSubmitDecision,
  onRefresh
}) => {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('fields');
  const [currentPage, setCurrentPage] = useState(1);
  const [highlightBox, setHighlightBox] = useState<BoundingBox | null>(null);
  
  // Correction Modal State
  const [correctionField, setCorrectionField] = useState<ExtractedFieldItem | null>(null);
  const [isCorrectionOpen, setIsCorrectionOpen] = useState(false);

  // Export Modal State
  const [isExportOpen, setIsExportOpen] = useState(false);

  const docId = document.document_id || document.id || '';

  // Bounding box jump handler
  const handleHighlight = (box: BoundingBox, pageNumber?: number) => {
    if (pageNumber && pageNumber !== currentPage) {
      setCurrentPage(pageNumber);
    }
    setHighlightBox(box);
  };

  // Correction click handler
  const handleStartCorrection = (field: ExtractedFieldItem) => {
    setCorrectionField(field);
    setIsCorrectionOpen(true);
  };

  const handleSaveCorrectionSubmit = async (payload: FieldCorrectionPayload) => {
    await onSaveCorrection(docId, payload);
    setIsCorrectionOpen(false);
    setCorrectionField(null);
    onRefresh();
  };

  const handleDecisionSubmit = async (payload: ReviewDecisionPayload) => {
    await onSubmitDecision(docId, payload);
    onRefresh();
  };

  const statusUpper = (document.status || '').toUpperCase();
  const isUploaded = statusUpper === 'UPLOADED';
  const isProcessing = statusUpper === 'PROCESSING';
  const isFailed = statusUpper === 'FAILED';
  const hasProcessed = !isUploaded && !isProcessing;

  const fields = document.extraction_results?.fields || [];
  const tables = document.table_results?.tables || [];
  const issues = document.validation_results?.issues || [];
  const confidenceScore = document.confidence_results?.composite_score ?? document.confidence ?? 0;
  const auditEntries = document.audit_trail || [];

  return (
    <div className="h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Workspace Subheader */}
      <div className="h-14 bg-slate-900 border-b border-slate-800 px-6 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={onBack}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            title="Back to list"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>

          <div className="h-4 w-px bg-slate-800" />

          <div className="flex items-center gap-2 min-w-0">
            <FileText className="w-4 h-4 text-slate-400 shrink-0" />
            <span className="font-semibold text-xs text-white truncate max-w-xs sm:max-w-md">
              {document.filename}
            </span>
          </div>

          <StatusBadge status={document.status} />

          {document.review_priority && document.review_priority !== 'NONE' && (
            <ReviewPriorityBadge priority={document.review_priority} />
          )}
        </div>

        <div className="flex items-center gap-2">
          {(isUploaded || isFailed) && (
            <button
              onClick={() => onProcess(docId)}
              disabled={isLoading || isProcessing}
              className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs px-3 py-1.5 rounded-lg transition-all shadow-md shadow-emerald-600/20 disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{isFailed ? 'Retry Pipeline' : 'Process Pipeline'}</span>
            </button>
          )}

          {hasProcessed && (
            <button
              onClick={() => setIsExportOpen(true)}
              className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium text-xs px-3 py-1.5 rounded-lg transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Split Layout */}
      {hasProcessed ? (
        <div className="flex-1 flex flex-col md:flex-row min-h-0 overflow-hidden">
          {/* Left Pane: Interactive Document Canvas */}
          <div className="w-full md:w-1/2 lg:w-3/5 h-1/2 md:h-full border-r border-slate-800 flex flex-col min-h-0">
            <DocumentViewer
              documentId={docId}
              pageCount={document.page_count}
              currentPage={currentPage}
              onPageChange={setCurrentPage}
              visualizations={document.visualizations}
              hasOverlays={Boolean(
                document.pages?.some((p) => p.has_overlay) ||
                (document.visualizations?.overlay_annotations && document.visualizations.overlay_annotations.length > 0)
              )}
              highlightBox={highlightBox}
              onClearHighlight={() => setHighlightBox(null)}
            />
          </div>

          {/* Right Pane: Analysis & Intelligence Tabs */}
          <div className="w-full md:w-1/2 lg:w-2/5 h-1/2 md:h-full flex flex-col min-h-0 bg-slate-900/50">
            {/* Tab Navigation */}
            <div className="bg-slate-900 border-b border-slate-800 px-4 flex items-center gap-1 shrink-0 overflow-x-auto">
              <button
                onClick={() => setActiveTab('fields')}
                className={`flex items-center gap-1.5 px-3 py-3 border-b-2 font-medium text-xs whitespace-nowrap transition-colors ${
                  activeTab === 'fields'
                    ? 'border-indigo-500 text-indigo-400 font-semibold'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Fields ({fields.length})</span>
              </button>

              <button
                onClick={() => setActiveTab('tables')}
                className={`flex items-center gap-1.5 px-3 py-3 border-b-2 font-medium text-xs whitespace-nowrap transition-colors ${
                  activeTab === 'tables'
                    ? 'border-indigo-500 text-indigo-400 font-semibold'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <TableIcon className="w-3.5 h-3.5" />
                <span>Tables ({tables.length})</span>
              </button>

              <button
                onClick={() => setActiveTab('validation')}
                className={`flex items-center gap-1.5 px-3 py-3 border-b-2 font-medium text-xs whitespace-nowrap transition-colors ${
                  activeTab === 'validation'
                    ? 'border-indigo-500 text-indigo-400 font-semibold'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Validation ({issues.length})</span>
              </button>

              <button
                onClick={() => setActiveTab('confidence')}
                className={`flex items-center gap-1.5 px-3 py-3 border-b-2 font-medium text-xs whitespace-nowrap transition-colors ${
                  activeTab === 'confidence'
                    ? 'border-indigo-500 text-indigo-400 font-semibold'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Confidence</span>
              </button>

              <button
                onClick={() => setActiveTab('audit')}
                className={`flex items-center gap-1.5 px-3 py-3 border-b-2 font-medium text-xs whitespace-nowrap transition-colors ${
                  activeTab === 'audit'
                    ? 'border-indigo-500 text-indigo-400 font-semibold'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <History className="w-3.5 h-3.5" />
                <span>Audit</span>
              </button>
            </div>

            {/* Tab Content Panels */}
            <div className="flex-1 min-h-0 overflow-y-auto">
              {activeTab === 'fields' && (
                <FieldExtractionPanel
                  fields={fields}
                  ocrText={document.ocr_text}
                  onHighlightField={handleHighlight}
                  onEditField={handleStartCorrection}
                />
              )}

              {activeTab === 'tables' && (
                <TableReviewPanel
                  tables={tables}
                  onHighlightCell={handleHighlight}
                />
              )}

              {activeTab === 'validation' && (
                <ValidationPanel
                  validationResults={document.validation_results}
                  onHighlightIssue={handleHighlight}
                />
              )}

              {activeTab === 'confidence' && (
                <div className="p-6">
                  <ConfidenceGauge
                    score={confidenceScore}
                    confidenceResults={document.confidence_results}
                  />
                </div>
              )}

              {activeTab === 'audit' && (
                <AuditTrailPanel
                  auditTrail={auditEntries}
                  corrections={document.corrections}
                  decisions={document.decisions}
                />
              )}
            </div>

            {/* Bottom Human Review Action Bar */}
            <ReviewDecisionBar
              status={document.status}
              onSubmitDecision={handleDecisionSubmit}
              isSubmitting={isLoading}
            />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-4">
          <div className="p-4 bg-slate-900 border border-slate-800 rounded-2xl text-slate-400">
            <FileText className={`w-12 h-12 text-indigo-400 ${isProcessing || isLoading ? 'animate-spin' : 'animate-pulse'}`} />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">
              {isProcessing || isLoading ? 'Processing Document...' : 'Document Not Processed'}
            </h2>
            <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
              {isProcessing || isLoading
                ? 'DocuMind AI is executing OCR, layout analysis, classification, and field extraction.'
                : 'This document has been uploaded but not yet passed through the DocuMind extraction & validation pipeline.'}
            </p>
          </div>

          {!isProcessing && !isLoading && (
            <button
              onClick={() => onProcess(docId)}
              disabled={isLoading}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/25"
            >
              <Play className="w-4 h-4" />
              <span>Run Processing Pipeline</span>
            </button>
          )}
        </div>
      )}

      {/* Field Correction Modal */}
      <CorrectionModal
        isOpen={isCorrectionOpen}
        field={correctionField}
        onClose={() => setIsCorrectionOpen(false)}
        onSave={handleSaveCorrectionSubmit}
      />

      {/* Export Download Modal */}
      <ExportModal
        isOpen={isExportOpen}
        documentId={docId}
        filename={document.filename}
        onClose={() => setIsExportOpen(false)}
      />
    </div>
  );
};
