import React from 'react';
import { ValidationIssue, ValidationReport, BoundingBox } from '../types';
import { StatusBadge } from './StatusBadge';
import { ShieldAlert, CheckCircle2, MapPin } from 'lucide-react';

interface Props {
  issues?: ValidationIssue[];
  validationResults?: ValidationReport;
  onHighlightIssue?: (bbox: BoundingBox, pageNumber?: number) => void;
  onSelectTarget?: (bbox?: BoundingBox, label?: string, pageNumber?: number) => void;
}

export const ValidationPanel: React.FC<Props> = ({
  issues: propIssues,
  validationResults,
  onHighlightIssue,
  onSelectTarget,
}) => {
  const issues = propIssues || validationResults?.issues || [];

  const handleHighlight = (bbox?: BoundingBox, label?: string, pageNumber?: number) => {
    if (!bbox) return;
    if (onHighlightIssue) {
      onHighlightIssue(bbox, pageNumber);
    } else if (onSelectTarget) {
      onSelectTarget(bbox, label, pageNumber);
    }
  };

  if (!issues || issues.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-slate-400 text-center bg-slate-900/60 border border-slate-800 rounded-xl m-4">
        <CheckCircle2 className="w-12 h-12 text-emerald-400 mb-3" />
        <h4 className="font-semibold text-slate-200 text-base">All Validation Rules Passed</h4>
        <p className="text-xs text-slate-400 mt-1 max-w-sm">
          No arithmetic discrepancies, format errors, or missing mandatory fields were identified.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-rose-400" />
          <h3 className="font-semibold text-slate-200 text-sm uppercase tracking-wider">
            Validation Findings & Consistency Rules
          </h3>
        </div>
        <span className="font-mono text-xs text-slate-400 bg-slate-800 px-2.5 py-1 rounded-md border border-slate-700">
          {issues.length} {issues.length === 1 ? 'finding' : 'findings'}
        </span>
      </div>

      <div className="space-y-3">
        {issues.map((issue, idx) => {
          const isCritical = issue.severity === 'critical' || issue.severity === 'CRITICAL';
          const isError = issue.severity === 'error' || issue.severity === 'ERROR';
          const isWarning = issue.severity === 'warning' || issue.severity === 'WARNING';
          const hasSpatial = !!issue.bounding_box;

          return (
            <div
              key={idx}
              className={`p-4 rounded-xl border transition-all ${
                isCritical
                  ? 'bg-rose-950/40 border-rose-600/80 shadow-rose-950/20'
                  : isError
                  ? 'bg-rose-950/20 border-rose-700/60'
                  : isWarning
                  ? 'bg-amber-950/20 border-amber-700/60'
                  : 'bg-slate-900 border-slate-800'
              }`}
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <StatusBadge type="severity" value={issue.severity} size="sm" />
                  <span className="font-mono text-xs font-semibold text-slate-300">
                    {issue.rule_id}
                  </span>
                </div>

                {hasSpatial ? (
                  <button
                    onClick={() =>
                      handleHighlight(issue.bounding_box, issue.rule_id, issue.page_number || 1)
                    }
                    className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition font-medium"
                    title="Highlight in Document"
                  >
                    <MapPin className="w-3.5 h-3.5" />
                    <span>Page {issue.page_number || 1}</span>
                  </button>
                ) : (
                  <span className="text-[11px] text-slate-500 italic font-mono">
                    [Unlocated / Non-Spatial]
                  </span>
                )}
              </div>

              {/* Message */}
              <div className="text-sm font-medium text-slate-100 mb-2">{issue.message}</div>

              {/* Suggested fix */}
              {issue.suggested_fix && (
                <div className="text-xs text-emerald-300 bg-emerald-950/40 border border-emerald-800/40 p-2 rounded mb-2">
                  <span className="font-semibold">Suggested Fix:</span> {issue.suggested_fix}
                </div>
              )}

              {/* Affected Fields & Category */}
              <div className="flex flex-wrap items-center gap-2 text-xs pt-2 border-t border-slate-800/80">
                {issue.category && (
                  <span className="text-slate-400 uppercase tracking-wider text-[10px] bg-slate-800 px-2 py-0.5 rounded">
                    Category: {issue.category}
                  </span>
                )}
                {issue.affected_fields && issue.affected_fields.length > 0 && (
                  <div className="flex items-center gap-1 text-slate-400 text-[11px]">
                    <span>Affected Fields:</span>
                    {issue.affected_fields.map((f, i) => (
                      <span
                        key={i}
                        className="font-mono text-indigo-300 bg-slate-800 px-1.5 py-0.5 rounded"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
