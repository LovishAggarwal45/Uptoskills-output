import React from 'react';
import { api } from '../api/client';
import { X, Download, FileJson, FileSpreadsheet, FileText } from 'lucide-react';

interface Props {
  isOpen?: boolean;
  documentId: string;
  filename: string;
  onClose: () => void;
}

export const ExportModal: React.FC<Props> = ({ isOpen = true, documentId, filename, onClose }) => {
  if (!isOpen) return null;

  const exportOptions = [
    {
      id: 'reviewed',
      title: 'Reviewed Structured JSON',
      desc: 'Complete extraction payload with human field corrections and immutable audit history.',
      icon: FileJson,
      badge: 'RECOMMENDED',
      color: 'border-indigo-500 bg-indigo-950/30 text-indigo-300',
      url: api.getExportUrl(documentId, 'reviewed'),
    },
    {
      id: 'json',
      title: 'Original Machine JSON',
      desc: 'Raw machine extraction results, OCR bounding boxes, and initial validation findings.',
      icon: FileJson,
      badge: 'RAW',
      color: 'border-slate-700 bg-slate-900 text-slate-300',
      url: api.getExportUrl(documentId, 'json'),
    },
    {
      id: 'csv',
      title: 'Tabular CSV Tables',
      desc: 'Extracted 2D tables, column headers, and line-item amounts in structured spreadsheet format.',
      icon: FileSpreadsheet,
      badge: 'CSV / ZIP',
      color: 'border-slate-700 bg-slate-900 text-slate-300',
      url: api.getExportUrl(documentId, 'csv'),
    },
    {
      id: 'report',
      title: 'Markdown Validation Report',
      desc: 'Executive summary with categorized rule evaluation tables and arithmetic checks.',
      icon: FileText,
      badge: 'MARKDOWN',
      color: 'border-slate-700 bg-slate-900 text-slate-300',
      url: api.getExportUrl(documentId, 'report'),
    },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl animate-in fade-in zoom-in duration-200">
        <div className="flex items-center justify-between px-6 py-4 bg-slate-800/80 border-b border-slate-700">
          <div className="flex items-center gap-2.5">
            <Download className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="font-semibold text-slate-100 text-base">Export Structured Results</h3>
              <p className="text-xs text-slate-400 font-mono truncate max-w-xs">{filename}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-3">
          {exportOptions.map((opt) => {
            const Icon = opt.icon;
            return (
              <a
                key={opt.id}
                href={opt.url}
                download
                className={`flex items-start justify-between gap-3 p-4 rounded-xl border transition-all hover:scale-[1.01] hover:shadow-lg block group ${opt.color}`}
              >
                <div className="flex items-start gap-3">
                  <div className="p-2.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 group-hover:border-indigo-500 transition">
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm text-slate-100">{opt.title}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 font-bold">
                        {opt.badge}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">{opt.desc}</p>
                  </div>
                </div>

                <Download className="w-4 h-4 text-slate-400 group-hover:text-indigo-400 transition mt-1 shrink-0" />
              </a>
            );
          })}
        </div>

        <div className="px-6 py-3 bg-slate-950/80 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
