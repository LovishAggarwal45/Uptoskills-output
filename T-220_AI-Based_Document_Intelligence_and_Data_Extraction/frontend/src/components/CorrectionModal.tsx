import React, { useState, useEffect } from 'react';
import { ExtractedField, FieldCorrectionPayload } from '../types';
import { api } from '../api/client';
import { X, Check, Edit3, AlertCircle } from 'lucide-react';

interface Props {
  isOpen?: boolean;
  documentId?: string;
  field: ExtractedField | null;
  onClose: () => void;
  onSave?: (payload: FieldCorrectionPayload) => Promise<void>;
  onSuccess?: () => void;
}

export const CorrectionModal: React.FC<Props> = ({
  isOpen = true,
  documentId,
  field,
  onClose,
  onSave,
  onSuccess,
}) => {
  const [correctedValue, setCorrectedValue] = useState<string>('');
  const [reason, setReason] = useState<string>('');
  const [reviewerId, setReviewerId] = useState<string>('senior_reviewer');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (field) {
      setCorrectedValue(field.value !== null && field.value !== undefined ? String(field.value) : '');
      setReason('');
      setError(null);
    }
  }, [field]);

  if (!isOpen || !field) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!correctedValue.trim()) {
      setError('Please provide a corrected value.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (onSave) {
        await onSave({
          field_name: field.name,
          original_value: field.value !== null ? String(field.value) : undefined,
          corrected_value: correctedValue.trim(),
          reviewer_id: reviewerId.trim() || 'senior_reviewer',
          reason: reason.trim() || undefined,
        });
      } else if (documentId) {
        await api.addCorrection(documentId, {
          field_name: field.name,
          original_value: field.value !== null ? String(field.value) : undefined,
          corrected_value: correctedValue.trim(),
          reviewer_id: reviewerId.trim() || 'senior_reviewer',
          reason: reason.trim() || undefined,
        });
        if (onSuccess) onSuccess();
      }
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to submit human correction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl animate-in fade-in zoom-in duration-200">
        <div className="flex items-center justify-between px-6 py-4 bg-slate-800/80 border-b border-slate-700">
          <div className="flex items-center gap-2.5">
            <Edit3 className="w-5 h-5 text-indigo-400" />
            <h3 className="font-semibold text-slate-100 text-base">
              Human Field Correction
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="p-3 bg-rose-950/80 border border-rose-700 rounded-lg text-rose-300 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Field Target
            </label>
            <input
              type="text"
              readOnly
              value={field.name}
              className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-300 font-mono text-sm cursor-not-allowed"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Original Extracted Value
            </label>
            <div className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-400 text-sm italic">
              {field.value !== null && field.value !== undefined
                ? String(field.value)
                : '(Missing / Not Extracted)'}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-200 uppercase tracking-wider mb-1">
              Corrected Value <span className="text-rose-400">*</span>
            </label>
            <input
              type="text"
              required
              value={correctedValue}
              onChange={(e) => setCorrectedValue(e.target.value)}
              placeholder="Enter verified correct value..."
              className="w-full px-3 py-2.5 bg-slate-950 border border-slate-700 rounded-lg text-slate-100 font-semibold text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              autoFocus
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                Reviewer ID
              </label>
              <input
                type="text"
                value={reviewerId}
                onChange={(e) => setReviewerId(e.target.value)}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-slate-300 text-xs focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                Reason / Note
              </label>
              <input
                type="text"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="e.g. OCR typo correction"
                className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-slate-300 text-xs focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800 flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-lg flex items-center gap-1.5 transition disabled:opacity-50"
            >
              {loading ? (
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Check className="w-4 h-4" />
              )}
              <span>Save Correction</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
