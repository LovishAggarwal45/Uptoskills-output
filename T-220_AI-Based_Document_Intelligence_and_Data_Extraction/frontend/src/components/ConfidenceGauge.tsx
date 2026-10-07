import React from 'react';
import { ConfidenceBand, DocumentConfidence } from '../types';
import { StatusBadge } from './StatusBadge';
import { ShieldCheck, Info } from 'lucide-react';

interface Props {
  score?: number;
  overallScore?: number;
  confidenceBand?: ConfidenceBand;
  confidenceResults?: DocumentConfidence;
  signals?: Record<string, number>;
  reviewRequired?: boolean;
}

export const ConfidenceGauge: React.FC<Props> = ({
  score,
  overallScore,
  confidenceBand,
  confidenceResults,
  signals: propSignals,
  reviewRequired: propReviewRequired,
}) => {
  const effectiveScore = score !== undefined ? score : (overallScore !== undefined ? overallScore : (confidenceResults?.composite_score ?? confidenceResults?.overall_confidence ?? 0));
  const percentage = Math.round(effectiveScore * 100);

  const effectiveBand = confidenceBand || confidenceResults?.confidence_band || (
    effectiveScore >= 0.85 ? 'HIGH' : effectiveScore >= 0.65 ? 'MEDIUM' : effectiveScore >= 0.4 ? 'LOW' : 'VERY_LOW'
  );

  const signals = propSignals || confidenceResults?.signals || {};
  const reviewRequired = propReviewRequired !== undefined ? propReviewRequired : (confidenceResults?.review_required || false);

  const getScoreColor = (sc: number) => {
    if (sc >= 0.85) return 'text-emerald-400 bg-emerald-500';
    if (sc >= 0.65) return 'text-blue-400 bg-blue-500';
    if (sc >= 0.4) return 'text-amber-400 bg-amber-500';
    return 'text-rose-400 bg-rose-500';
  };

  const signalLabels: Record<string, string> = {
    ocr_quality: 'OCR Optical Clarity',
    extraction_pattern: 'Pattern & Regex Fit',
    spatial_proximity: '2D Spatial Proximity',
    validation_integrity: 'Rule Validation Integrity',
    table_regularity: 'Tabular Grid Structure',
    classification_confidence: 'Document Classification',
    mean_field_confidence: 'Field Extraction Average',
    mean_table_confidence: 'Table Quality Average',
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-indigo-400" />
          <h3 className="font-bold text-slate-200 text-sm tracking-wide uppercase">
            Multi-Tier Confidence Scoring
          </h3>
        </div>
        <StatusBadge type="confidence" value={effectiveBand} size="sm" />
      </div>

      <div className="flex items-end gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <div className="text-4xl font-extrabold text-white font-mono tracking-tight">
            {percentage}%
          </div>
          <div className="text-xs text-slate-400 mt-0.5">Composite Score</div>
        </div>

        <div className="flex-1 pb-1">
          <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden p-0.5 border border-slate-700">
            <div
              className={`h-full rounded-full transition-all duration-500 ${getScoreColor(
                effectiveScore
              ).split(' ')[1]}`}
              style={{ width: `${Math.max(5, Math.min(100, percentage))}%` }}
            />
          </div>
        </div>
      </div>

      {reviewRequired && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-xs text-amber-300">
          Human verification is recommended based on confidence threshold routing.
        </div>
      )}

      {Object.keys(signals).length > 0 && (
        <div className="space-y-3">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Signal Weight Breakdown
          </div>
          {Object.entries(signals).map(([key, val]) => {
            const label = signalLabels[key] || key.replace(/_/g, ' ');
            const valPct = Math.round(Number(val) * 100);
            return (
              <div key={key} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 capitalize">{label}</span>
                  <span className="font-mono text-slate-400">{valPct}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-indigo-500 h-full rounded-full"
                    style={{ width: `${Math.max(2, Math.min(100, valPct))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="pt-3 border-t border-slate-800/60 flex items-start gap-2 text-xs text-slate-400">
        <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <span>
          Scores are mathematically aggregated across OCR engine confidence, field-level extraction models, and validation engine checks.
        </span>
      </div>
    </div>
  );
};
