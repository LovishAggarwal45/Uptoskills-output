import React, { useState } from 'react';
import { api } from '../api/client';
import { ReviewDecisionPayload } from '../types';
import { CheckCircle, AlertTriangle, ShieldCheck, Send } from 'lucide-react';

interface Props {
  documentId?: string;
  status?: string;
  currentStatus?: string;
  hasCriticalIssues?: boolean;
  onSubmitDecision?: (payload: ReviewDecisionPayload) => Promise<void>;
  onDecisionSubmitted?: () => void;
  isSubmitting?: boolean;
}

export const ReviewDecisionBar: React.FC<Props> = ({
  documentId,
  status,
  currentStatus: propStatus,
  hasCriticalIssues = false,
  onSubmitDecision,
  onDecisionSubmitted,
  isSubmitting = false,
}) => {
  const currentStatus = status || propStatus || 'PENDING_REVIEW';
  const [showDialog, setShowDialog] = useState<boolean>(false);
  const [selectedAction, setSelectedAction] = useState<string>('approve');
  const [reviewerNotes, setReviewerNotes] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleActionClick = (action: string) => {
    setSelectedAction(action);
    setShowDialog(true);
  };

  const handleConfirmSubmit = async () => {
    setLoading(true);
    try {
      if (onSubmitDecision) {
        await onSubmitDecision({
          action: selectedAction,
          reviewer_id: 'lead_reviewer',
          notes: reviewerNotes.trim() || undefined,
        });
      } else if (documentId) {
        await api.submitReviewDecision(documentId, {
          action: selectedAction,
          reviewer_id: 'lead_reviewer',
          notes: reviewerNotes.trim() || undefined,
        });
        if (onDecisionSubmitted) onDecisionSubmitted();
      }
      setShowDialog(false);
      setReviewerNotes('');
    } catch (err: any) {
      alert(`Decision submission failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const isBusy = loading || isSubmitting;

  return (
    <>
      <div className="bg-slate-900/95 border-t border-slate-800 p-4 shadow-xl flex flex-wrap items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-indigo-400" />
          <div>
            <div className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
              Human Review & Decision Actions
            </div>
            <div className="text-xs text-slate-400">
              Current Review Status:{' '}
              <span className="font-mono text-slate-200 capitalize font-medium">{currentStatus}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => handleActionClick('confirm')}
            disabled={isBusy}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition disabled:opacity-50"
          >
            Confirm Extracted
          </button>

          {hasCriticalIssues ? (
            <button
              onClick={() => handleActionClick('override')}
              disabled={isBusy}
              className="px-3 py-1.5 bg-amber-950/80 hover:bg-amber-900/80 text-amber-300 text-xs font-semibold rounded-lg border border-amber-700 transition flex items-center gap-1.5 disabled:opacity-50"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Override with Justification</span>
            </button>
          ) : (
            <button
              onClick={() => handleActionClick('approve')}
              disabled={isBusy}
              className="px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-semibold rounded-lg shadow-lg transition flex items-center gap-1.5 disabled:opacity-50"
            >
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Approve Document</span>
            </button>
          )}

          <button
            onClick={() => handleActionClick('reject')}
            disabled={isBusy}
            className="px-3 py-1.5 bg-rose-950/80 hover:bg-rose-900/80 text-rose-300 text-xs font-semibold rounded-lg border border-rose-800 transition disabled:opacity-50"
          >
            Reject Document
          </button>
        </div>
      </div>

      {/* Confirmation Dialog */}
      {showDialog && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl p-6 space-y-4">
            <h3 className="font-semibold text-slate-100 text-base flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-indigo-400" />
              <span>Confirm Review Decision: {selectedAction.toUpperCase()}</span>
            </h3>

            <p className="text-xs text-slate-400">
              Submitting this action will update the formal review status in the database and append a verified entry to the audit log.
            </p>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Reviewer Rationale / Notes
              </label>
              <textarea
                value={reviewerNotes}
                onChange={(e) => setReviewerNotes(e.target.value)}
                placeholder="Enter justification or sign-off notes..."
                rows={3}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-slate-200 text-xs focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-2">
              <button
                type="button"
                onClick={() => setShowDialog(false)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmSubmit}
                disabled={isBusy}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 shadow-lg disabled:opacity-50"
              >
                {isBusy ? (
                  <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="w-3.5 h-3.5" />
                )}
                <span>Submit Decision</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
