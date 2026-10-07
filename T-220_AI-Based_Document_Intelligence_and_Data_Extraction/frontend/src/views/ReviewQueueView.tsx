import React from 'react';
import { 
  ArrowRight, 
  FileText, 
  CheckCircle2, 
  RefreshCw,
  Clock
} from 'lucide-react';
import { DocumentItem } from '../types';
import { ReviewPriorityBadge } from '../components/StatusBadge';

interface ReviewQueueViewProps {
  documents: DocumentItem[];
  isLoading: boolean;
  onSelectDocument: (docId: string) => void;
  onRefresh: () => void;
}

export const ReviewQueueView: React.FC<ReviewQueueViewProps> = ({
  documents,
  isLoading,
  onSelectDocument,
  onRefresh
}) => {
  // Filter only documents requiring human review or pending review
  const reviewDocs = documents.filter(
    (d) => d.status === 'PENDING_REVIEW' || d.status === 'needs_review' || (d.review_priority && d.review_priority.toString().toUpperCase() !== 'NONE')
  );

  // Sort by priority severity: CRITICAL > HIGH > MEDIUM > LOW
  const priorityWeight: Record<string, number> = {
    CRITICAL: 4,
    URGENT: 4,
    HIGH: 3,
    MEDIUM: 2,
    LOW: 1,
    NONE: 0
  };

  const sortedDocs = [...reviewDocs].sort((a, b) => {
    const pA = priorityWeight[a.review_priority?.toString().toUpperCase() || 'NONE'] || 0;
    const pB = priorityWeight[b.review_priority?.toString().toUpperCase() || 'NONE'] || 0;
    return pB - pA;
  });

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto overflow-y-auto max-h-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">Human Review Queue</h1>
            <span className="bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2.5 py-0.5 rounded-full text-xs font-bold">
              {reviewDocs.length} Pending
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Documents routed for human verification due to validation errors, arithmetic discrepancies, or confidence thresholds.
          </p>
        </div>

        <button
          onClick={onRefresh}
          className="p-2 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl transition-all self-start sm:self-auto"
          title="Refresh Queue"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Review Queue Cards */}
      {sortedDocs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sortedDocs.map((doc) => {
            const prioUpper = (doc.review_priority || '').toString().toUpperCase();
            const isCritical = prioUpper === 'CRITICAL' || prioUpper === 'URGENT';
            const isHigh = prioUpper === 'HIGH';
            const docId = doc.id || doc.document_id || '';
            const conf = doc.confidence ?? doc.confidence_score;

            return (
              <div
                key={docId}
                className={`bg-slate-900/90 border rounded-2xl p-5 shadow-lg transition-all hover:border-slate-700 flex flex-col justify-between ${
                  isCritical 
                    ? 'border-rose-500/40 bg-rose-950/10' 
                    : isHigh 
                    ? 'border-amber-500/40 bg-amber-950/10' 
                    : 'border-slate-800'
                }`}
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="p-2.5 bg-slate-800 rounded-xl text-slate-300 border border-slate-700">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-sm font-bold text-slate-100 truncate">{doc.filename}</h3>
                        <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                          <span className="capitalize">{doc.document_type ? doc.document_type.replace('_', ' ') : 'Unclassified'}</span>
                          <span>·</span>
                          <span>{doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}</span>
                        </div>
                      </div>
                    </div>

                    <ReviewPriorityBadge priority={doc.review_priority || 'LOW'} />
                  </div>

                  {/* Summary Bar */}
                  <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex items-center justify-between text-xs">
                    <div className="space-y-0.5">
                      <span className="text-[10px] uppercase font-semibold text-slate-500">Confidence</span>
                      <p className="font-mono font-bold text-slate-200">
                        {conf !== undefined && conf !== null ? `${(conf * 100).toFixed(0)}%` : '—'}
                      </p>
                    </div>

                    <div className="space-y-0.5">
                      <span className="text-[10px] uppercase font-semibold text-slate-500">Status</span>
                      <p className="font-semibold text-amber-400">
                        {doc.status.replace('_', ' ')}
                      </p>
                    </div>

                    <div className="space-y-0.5">
                      <span className="text-[10px] uppercase font-semibold text-slate-500">Ingested</span>
                      <p className="font-mono text-slate-400 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(doc.created_at || doc.uploaded_at || Date.now()).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 mt-3 border-t border-slate-800/60 flex items-center justify-between">
                  <span className="text-[11px] text-slate-500 font-mono">
                    ID: {docId.slice(0, 12)}...
                  </span>

                  <button
                    onClick={() => onSelectDocument(docId)}
                    className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-xl transition-all shadow-md shadow-indigo-600/20"
                  >
                    <span>Start Review</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center justify-center mx-auto">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <h3 className="text-base font-semibold text-slate-200">Review Queue is Empty</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            All processed documents either met straight-through processing criteria or have already been reviewed and approved.
          </p>
        </div>
      )}
    </div>
  );
};
