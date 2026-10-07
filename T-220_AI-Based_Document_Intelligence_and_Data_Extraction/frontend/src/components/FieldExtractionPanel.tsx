import React from 'react';
import { ExtractedField, BoundingBox } from '../types';
import { StatusBadge } from './StatusBadge';
import {
  Edit3,
  MapPin,
  HelpCircle,
  Hash,
  Calendar,
  DollarSign,
  User,
  Info,
} from 'lucide-react';

interface Props {
  fields: ExtractedField[] | Record<string, ExtractedField>;
  ocrText?: string;
  onHighlightField?: (bbox: BoundingBox, pageNumber?: number) => void;
  onSelectTarget?: (bbox?: BoundingBox, label?: string, pageNumber?: number) => void;
  onEditField?: (field: ExtractedField) => void;
  onOpenCorrection?: (field: ExtractedField) => void;
}

export const FieldExtractionPanel: React.FC<Props> = ({
  fields: rawFields,
  ocrText,
  onHighlightField,
  onSelectTarget,
  onEditField,
  onOpenCorrection,
}) => {
  const handleHighlight = (bbox?: BoundingBox, label?: string, pageNumber?: number) => {
    if (!bbox) return;
    if (onHighlightField) {
      onHighlightField(bbox, pageNumber);
    } else if (onSelectTarget) {
      onSelectTarget(bbox, label, pageNumber);
    }
  };

  const handleEdit = (field: ExtractedField) => {
    if (onEditField) {
      onEditField(field);
    } else if (onOpenCorrection) {
      onOpenCorrection(field);
    }
  };

  // Normalize fields into an array
  const fieldList: ExtractedField[] = Array.isArray(rawFields)
    ? rawFields
    : Object.entries(rawFields || {}).map(([name, f]) => ({ ...f, name: f.name || name }));

  if (!fieldList || fieldList.length === 0) {
    return (
      <div className="p-6 space-y-4">
        <div className="flex flex-col items-center justify-center p-6 text-slate-400 text-center bg-slate-900/60 rounded-xl border border-slate-800">
          <HelpCircle className="w-8 h-8 text-indigo-400 mb-2" />
          <h4 className="font-semibold text-slate-200 text-sm">No Structured Invoice Fields</h4>
          <p className="text-xs text-slate-400 mt-1 max-w-sm">
            This document type (e.g. resume or general document) does not contain tabular financial invoice keys. Optical Character Recognition (OCR) succeeded.
          </p>
        </div>

        {ocrText && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 pb-1 border-b border-slate-800">
              <span className="flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-indigo-400" />
                <span>Extracted OCR Text</span>
              </span>
              <span className="font-mono text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                {ocrText.length} chars
              </span>
            </div>
            <div className="p-4 bg-slate-900/80 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed max-h-[calc(100vh-320px)] overflow-y-auto select-text">
              {ocrText}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Categorize fields
  const categorizeField = (name: string) => {
    const n = name.toLowerCase();
    if (n.includes('number') || n.includes('id') || n.includes('code') || n.includes('po')) return 'Identifiers';
    if (n.includes('date') || n.includes('time') || n.includes('dob')) return 'Dates & Timing';
    if (n.includes('total') || n.includes('subtotal') || n.includes('tax') || n.includes('amount') || n.includes('price')) return 'Financial Amounts';
    if (n.includes('vendor') || n.includes('merchant') || n.includes('customer') || n.includes('applicant') || n.includes('name')) return 'Parties & Names';
    return 'General Fields';
  };

  const categories: Record<string, ExtractedField[]> = {};
  fieldList.forEach((fld) => {
    const cat = categorizeField(fld.name);
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(fld);
  });

  const getCategoryIcon = (cat: string) => {
    switch (cat) {
      case 'Identifiers': return <Hash className="w-4 h-4 text-indigo-400" />;
      case 'Dates & Timing': return <Calendar className="w-4 h-4 text-emerald-400" />;
      case 'Financial Amounts': return <DollarSign className="w-4 h-4 text-teal-400" />;
      case 'Parties & Names': return <User className="w-4 h-4 text-violet-400" />;
      default: return <Info className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <div className="p-4 space-y-6">
      {Object.entries(categories).map(([category, items]) => (
        <div key={category} className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 pb-1 border-b border-slate-800">
            {getCategoryIcon(category)}
            <span>{category}</span>
            <span className="ml-auto font-mono text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">
              {items.length}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-2.5">
            {items.map((fld) => {
              const confPct = Math.round((fld.extraction_confidence ?? 0) * 100);
              const hasProvenance = !!fld.provenance?.bounding_box;

              return (
                <div
                  key={fld.name}
                  className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 hover:border-slate-700 transition-all shadow-sm group"
                >
                  <div className="flex items-start justify-between gap-2 mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-semibold text-slate-300">
                        {fld.name}
                      </span>
                      {fld.is_human_corrected && (
                        <span className="bg-amber-950 border border-amber-600 text-amber-300 font-mono text-[10px] px-1.5 py-0.5 rounded">
                          [CORRECTED]
                        </span>
                      )}
                      {fld.is_required && (
                        <span className="text-rose-400 text-[10px] font-bold" title="Required Field">*</span>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5">
                      {fld.validation_status && (
                        <StatusBadge type="validation" value={fld.validation_status} size="sm" showIcon={false} />
                      )}
                      <button
                        onClick={() => handleEdit(fld)}
                        className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition"
                        title="Propose Field Correction"
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Value Row */}
                  <div className="flex items-baseline justify-between gap-3 mt-1">
                    <div className="text-sm font-semibold text-slate-100 break-words flex-1">
                      {fld.value !== null && fld.value !== undefined ? (
                        <span>{String(fld.value)}</span>
                      ) : (
                        <span className="text-slate-500 italic text-xs">Missing / Not Detected</span>
                      )}
                    </div>

                    {/* Normalized value preview */}
                    {fld.normalized_value !== undefined && fld.normalized_value !== fld.value && (
                      <span className="font-mono text-xs text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/60">
                        Norm: {String(fld.normalized_value)}
                      </span>
                    )}
                  </div>

                  {/* Confidence Bar & Provenance Trigger */}
                  <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[11px] text-slate-400">Conf: {confPct}%</span>
                      <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            confPct >= 85 ? 'bg-emerald-500' : confPct >= 65 ? 'bg-blue-500' : 'bg-amber-500'
                          }`}
                          style={{ width: `${confPct}%` }}
                        />
                      </div>
                    </div>

                    {hasProvenance ? (
                      <button
                        onClick={() =>
                          handleHighlight(
                            fld.provenance?.bounding_box,
                            fld.name,
                            fld.provenance?.page_number
                          )
                        }
                        className="flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 transition"
                      >
                        <MapPin className="w-3 h-3" />
                        <span>Page {fld.provenance?.page_number || 1}</span>
                      </button>
                    ) : (
                      <span className="text-[11px] text-slate-500 italic">No spatial bbox</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
};
