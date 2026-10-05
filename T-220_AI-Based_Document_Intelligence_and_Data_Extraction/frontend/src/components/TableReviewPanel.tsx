import React from 'react';
import { Table, BoundingBox } from '../types';
import { Table as TableIcon, AlertTriangle, CheckCircle2, MapPin, Layers } from 'lucide-react';

interface Props {
  tables: Table[];
  onHighlightCell?: (bbox: BoundingBox, pageNumber?: number) => void;
  onSelectTarget?: (bbox?: BoundingBox, label?: string, pageNumber?: number) => void;
}

export const TableReviewPanel: React.FC<Props> = ({ tables, onHighlightCell, onSelectTarget }) => {
  const handleHighlight = (bbox?: BoundingBox, label?: string, pageNumber?: number) => {
    if (!bbox) return;
    if (onHighlightCell) {
      onHighlightCell(bbox, pageNumber);
    } else if (onSelectTarget) {
      onSelectTarget(bbox, label, pageNumber);
    }
  };

  if (!tables || tables.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-slate-400 text-center">
        <TableIcon className="w-10 h-10 text-slate-600 mb-3" />
        <h4 className="font-semibold text-slate-300">No Tabular Structures Detected</h4>
        <p className="text-xs text-slate-500 mt-1 max-w-xs">
          The document does not contain multi-column grids or line-item tables.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      {tables.map((table, tIdx) => {
        const lineItems = table.line_items || [];
        const hasMathError = lineItems.some((li) => !li.is_valid_arithmetic);

        return (
          <div
            key={table.table_id || tIdx}
            className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-lg"
          >
            {/* Table Header Card */}
            <div className="flex items-center justify-between px-4 py-3 bg-slate-800/80 border-b border-slate-700">
              <div className="flex items-center gap-2">
                <TableIcon className="w-4 h-4 text-teal-400" />
                <span className="font-semibold text-sm text-slate-200">
                  {table.table_id || `Table ${tIdx + 1}`}
                </span>
                <span className="text-xs font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                  Page {table.page_number}
                </span>
              </div>

              <div className="flex items-center gap-3">
                {hasMathError ? (
                  <span className="inline-flex items-center gap-1 bg-rose-950/80 border border-rose-600 text-rose-300 text-xs px-2 py-0.5 rounded font-mono">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    [MATH-ERR]
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 bg-emerald-950/80 border border-emerald-700 text-emerald-300 text-xs px-2 py-0.5 rounded font-mono">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    [MATH-OK]
                  </span>
                )}

                {table.bounding_box && (
                  <button
                    onClick={() =>
                      handleHighlight(table.bounding_box, table.table_id, table.page_number)
                    }
                    className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-700 transition"
                    title="Highlight Table in Document"
                  >
                    <MapPin className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Render Tabular Data Grid */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-200">
                <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="px-3 py-2.5 w-10 text-center font-mono">#</th>
                    {table.headers && table.headers.length > 0 ? (
                      table.headers.map((h, i) => (
                        <th key={i} className="px-3 py-2.5">
                          {h}
                        </th>
                      ))
                    ) : (
                      <th className="px-3 py-2.5">Item Details</th>
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {table.rows && table.rows.length > 0 ? (
                    table.rows.map((row, rIdx) => (
                      <tr
                        key={rIdx}
                        onClick={() =>
                          row.bounding_box &&
                          handleHighlight(row.bounding_box, `Row ${rIdx + 1}`, table.page_number)
                        }
                        className="hover:bg-slate-800/50 transition cursor-pointer"
                      >
                        <td className="px-3 py-2 text-center text-slate-500">{rIdx + 1}</td>
                        {row.cells.map((cell, cIdx) => (
                          <td key={cIdx} className="px-3 py-2 text-slate-200">
                            {cell.text || '-'}
                          </td>
                        ))}
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td
                        colSpan={Math.max(2, (table.headers || []).length + 1)}
                        className="px-4 py-3 text-center text-slate-500 italic"
                      >
                        No data rows parsed for this table.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Line-item Math Reconciliations */}
            {lineItems.length > 0 && (
              <div className="p-3 bg-slate-950/60 border-t border-slate-800 space-y-2">
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Line-Item Arithmetic Reconciliations</span>
                </div>
                <div className="space-y-1.5">
                  {lineItems.map((li, i) => {
                    const mathOk = li.is_valid_arithmetic;
                    return (
                      <div
                        key={i}
                        className={`flex items-center justify-between text-xs px-3 py-1.5 rounded border ${
                          mathOk
                            ? 'bg-slate-900 border-slate-800 text-slate-300'
                            : 'bg-rose-950/60 border-rose-700 text-rose-200'
                        }`}
                      >
                        <div className="flex items-center gap-2 truncate flex-1 mr-2">
                          <span className="font-mono text-slate-400 text-[10px]">#{li.row_index + 1}</span>
                          <span className="truncate">{li.description || 'Line item'}</span>
                        </div>

                        <div className="flex items-center gap-3 font-mono text-[11px] shrink-0">
                          <span>
                            {li.quantity ?? 1} × ${li.unit_price ?? 0} = ${li.amount ?? 0}
                          </span>
                          {!mathOk && (
                            <span className="text-rose-400 font-bold text-[10px] bg-rose-900/60 px-1.5 py-0.5 rounded">
                              MISMATCH
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
