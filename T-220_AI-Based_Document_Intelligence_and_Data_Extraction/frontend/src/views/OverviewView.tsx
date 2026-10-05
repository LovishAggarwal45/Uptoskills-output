import React from 'react';
import { 
  FileText, 
  AlertCircle, 
  TrendingUp, 
  Clock, 
  ArrowRight, 
  ShieldAlert, 
  BarChart3, 
  Layers, 
  Cpu, 
  CheckCircle2 
} from 'lucide-react';
import { SystemStats, DocumentItem } from '../types';
import { StatusBadge, ReviewPriorityBadge } from '../components/StatusBadge';

interface OverviewViewProps {
  stats: SystemStats | null;
  documents: DocumentItem[];
  onSelectDocument: (docId: string) => void;
  onNavigateToReview: () => void;
  onNavigateToDocuments: () => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  stats,
  documents,
  onSelectDocument,
  onNavigateToReview,
  onNavigateToDocuments
}) => {
  const total = stats?.total_documents || 0;
  const processed = stats?.completed_documents || stats?.processed_documents || 0;
  const pendingReview = stats?.awaiting_review || stats?.pending_review_documents || 0;
  const approved = stats?.approved_documents || 0;
  const stpRate = stats?.straight_through_processing_rate_pct || (total > 0 ? ((total - pendingReview) / total) * 100 : 0);
  const avgConfidence = stats?.average_confidence_pct || 0;

  const typeDistribution = stats?.document_type_distribution || stats?.by_type || {};
  const priorityBreakdown = stats?.priority_breakdown || stats?.by_priority || {};

  const recentDocs = documents.slice(0, 5);

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto overflow-y-auto max-h-full">
      {/* Top Welcome & KPI Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Intelligence Dashboard</h1>
          <p className="text-sm text-slate-400">
            Real-time multi-tier processing metrics, straight-through automation rates, and human review routing.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onNavigateToReview}
            className="flex items-center gap-2 bg-amber-500/15 hover:bg-amber-500/25 text-amber-400 border border-amber-500/30 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all"
          >
            <ShieldAlert className="w-4 h-4" />
            <span>Review Queue ({pendingReview})</span>
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Documents</span>
            <div className="p-2 bg-indigo-500/10 rounded-xl text-indigo-400 border border-indigo-500/20">
              <FileText className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white font-mono">{total}</span>
            <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
              <span className="text-emerald-400 font-medium">{processed}</span> processed
            </div>
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">STP Automation Rate</span>
            <div className="p-2 bg-emerald-500/10 rounded-xl text-emerald-400 border border-emerald-500/20">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white font-mono">{stpRate.toFixed(1)}%</span>
            <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
              <span>Straight-Through Processing</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Pending Review</span>
            <div className="p-2 bg-amber-500/10 rounded-xl text-amber-400 border border-amber-500/20">
              <AlertCircle className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-amber-400 font-mono">{pendingReview}</span>
            <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
              <span className="text-emerald-400 font-medium">{approved}</span> approved
            </div>
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Avg Confidence</span>
            <div className="p-2 bg-blue-500/10 rounded-xl text-blue-400 border border-blue-500/20">
              <BarChart3 className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold text-white font-mono">
              {avgConfidence > 0 ? `${avgConfidence.toFixed(1)}%` : '—'}
            </span>
            <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
              <span>Weighted Multi-Tier Score</span>
            </div>
          </div>
        </div>
      </div>

      {/* Subsystem & Review Priority Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Document Classification Distribution */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              Document Class Distribution
            </h3>
          </div>

          <div className="space-y-3">
            {Object.keys(typeDistribution).length > 0 ? (
              Object.entries(typeDistribution).map(([type, countRaw]) => {
                const count = Number(countRaw) || 0;
                const pct = total > 0 ? (count / total) * 100 : 0;
                return (
                  <div key={type} className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="text-slate-300 capitalize">{type.replace('_', ' ')}</span>
                      <span className="text-slate-400 font-mono">{count} ({pct.toFixed(0)}%)</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="text-xs text-slate-500 italic py-4 text-center">No documents classified yet.</p>
            )}
          </div>
        </div>

        {/* Human Review Priority Breakdown */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-400" />
              Review Priority Breakdown
            </h3>
          </div>

          <div className="space-y-3">
            {Object.keys(priorityBreakdown).length > 0 ? (
              Object.entries(priorityBreakdown).map(([priority, countRaw]) => {
                const count = Number(countRaw) || 0;
                const colors: Record<string, string> = {
                  CRITICAL: 'bg-rose-500',
                  URGENT: 'bg-rose-500',
                  HIGH: 'bg-amber-500',
                  MEDIUM: 'bg-yellow-500',
                  LOW: 'bg-blue-500',
                  NONE: 'bg-emerald-500'
                };
                const color = colors[priority.toUpperCase()] || 'bg-slate-500';
                const pct = total > 0 ? (count / total) * 100 : 0;

                return (
                  <div key={priority} className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="text-slate-300">{priority}</span>
                      <span className="text-slate-400 font-mono">{count}</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${color} rounded-full transition-all duration-500`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="text-xs text-slate-500 italic py-4 text-center">No review priority data.</p>
            )}
          </div>
        </div>

        {/* Pipeline Intelligence Architecture Status */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-400" />
              Subsystem Pipeline
            </h3>
          </div>

          <div className="space-y-2.5 text-xs">
            {[
              { phase: 'Phases 1–3', label: 'OCR & Ingestion', status: 'Ready' },
              { phase: 'Phase 4', label: 'Deterministic Classification', status: 'Active' },
              { phase: 'Phases 5–6', label: 'Field & Table Intelligence', status: 'Active' },
              { phase: 'Phase 7', label: 'Validation & Consistency Engine', status: 'Active' },
              { phase: 'Phase 8', label: 'Multi-Tier Confidence Routing', status: 'Active' },
              { phase: 'Phases 9–10', label: 'Visual Explainability & Dashboard', status: 'Online' },
            ].map((p, idx) => (
              <div key={idx} className="flex items-center justify-between py-1 border-b border-slate-800/60 last:border-0">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono font-bold text-slate-500">{p.phase}</span>
                  <span className="text-slate-300 font-medium">{p.label}</span>
                </div>
                <span className="text-emerald-400 font-medium flex items-center gap-1 text-[11px]">
                  <CheckCircle2 className="w-3 h-3" /> {p.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-slate-400" />
            <h3 className="text-sm font-semibold text-white">Recent Documents</h3>
          </div>
          <button
            onClick={onNavigateToDocuments}
            className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
          >
            <span>View All Library</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {recentDocs.length > 0 ? (
          <div className="divide-y divide-slate-800/80">
            {recentDocs.map((doc) => {
              const docId = doc.id || doc.document_id || '';
              const conf = doc.confidence ?? doc.confidence_score;

              return (
                <div
                  key={docId}
                  onClick={() => onSelectDocument(docId)}
                  className="p-4 hover:bg-slate-800/40 cursor-pointer transition-colors flex items-center justify-between gap-4"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="p-2 bg-slate-800 rounded-lg text-slate-400 border border-slate-700">
                      <FileText className="w-4 h-4" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-slate-200 truncate">{doc.filename}</p>
                      <div className="flex items-center gap-2 mt-0.5 text-[11px] text-slate-400">
                        <span className="capitalize">{doc.document_type ? doc.document_type.replace('_', ' ') : 'Unclassified'}</span>
                        <span>·</span>
                        <span>{doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}</span>
                        <span>·</span>
                        <span>{new Date(doc.created_at || doc.uploaded_at || Date.now()).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    {doc.review_priority && doc.review_priority.toString().toUpperCase() !== 'NONE' && (
                      <ReviewPriorityBadge priority={doc.review_priority} />
                    )}
                    <StatusBadge status={doc.status} />
                    <div className="text-right">
                      <span className="text-xs font-mono font-bold text-slate-200">
                        {conf !== undefined && conf !== null ? `${(conf * 100).toFixed(0)}%` : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-8 text-center text-slate-500 text-xs">
            No documents uploaded yet. Click "Upload Document" to begin.
          </div>
        )}
      </div>
    </div>
  );
};
