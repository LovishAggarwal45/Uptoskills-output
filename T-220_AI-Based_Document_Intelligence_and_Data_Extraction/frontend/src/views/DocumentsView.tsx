import React, { useState, useMemo } from 'react';
import { 
  FileText, 
  Search, 
  Filter, 
  Play, 
  Trash2, 
  ExternalLink, 
  ChevronLeft, 
  ChevronRight,
  RefreshCw,
  Plus
} from 'lucide-react';
import { DocumentItem } from '../types';
import { StatusBadge, ReviewPriorityBadge } from '../components/StatusBadge';

interface DocumentsViewProps {
  documents: DocumentItem[];
  isLoading: boolean;
  onSelectDocument: (docId: string) => void;
  onProcessDocument: (docId: string) => void;
  onDeleteDocument: (docId: string) => void;
  onOpenUpload: () => void;
  onRefresh: () => void;
}

export const DocumentsView: React.FC<DocumentsViewProps> = ({
  documents,
  isLoading,
  onSelectDocument,
  onProcessDocument,
  onDeleteDocument,
  onOpenUpload,
  onRefresh
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [priorityFilter, setPriorityFilter] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Extract unique document types
  const docTypes = useMemo(() => {
    const types = new Set<string>();
    documents.forEach((d) => {
      if (d.document_type) types.add(d.document_type);
    });
    return Array.from(types);
  }, [documents]);

  // Filtered documents
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      const docId = doc.id || doc.document_id || '';
      // Search
      if (searchTerm) {
        const query = searchTerm.toLowerCase();
        const matchName = doc.filename.toLowerCase().includes(query);
        const matchType = (doc.document_type || '').toLowerCase().includes(query);
        const matchId = docId.toLowerCase().includes(query);
        if (!matchName && !matchType && !matchId) return false;
      }

      // Status
      if (statusFilter !== 'ALL' && doc.status?.toUpperCase() !== statusFilter.toUpperCase()) {
        return false;
      }

      // Document Type
      if (typeFilter !== 'ALL' && doc.document_type?.toUpperCase() !== typeFilter.toUpperCase()) {
        return false;
      }

      // Review Priority
      if (priorityFilter !== 'ALL' && doc.review_priority?.toString().toUpperCase() !== priorityFilter.toUpperCase()) {
        return false;
      }

      return true;
    });
  }, [documents, searchTerm, statusFilter, typeFilter, priorityFilter]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredDocuments.length / pageSize));
  const paginatedDocuments = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredDocuments.slice(start, start + pageSize);
  }, [filteredDocuments, currentPage, pageSize]);

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto overflow-y-auto max-h-full">
      {/* View Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Document Library</h1>
          <p className="text-sm text-slate-400">
            Manage, process, and inspect all ingested PDF and image files in the DocuMind platform.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onRefresh}
            className="p-2 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl transition-all"
            title="Refresh list"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-3.5 py-2 rounded-xl transition-all shadow-lg shadow-indigo-600/20"
          >
            <Plus className="w-4 h-4" />
            <span>Upload New</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-2xl flex flex-wrap items-center gap-3 shadow-md">
        {/* Search input */}
        <div className="relative flex-1 min-w-[220px]">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search by filename, type, or ID..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full bg-slate-950/70 border border-slate-700/80 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
          />
        </div>

        {/* Status filter */}
        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-slate-500 hidden sm:block" />
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="bg-slate-950/70 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
          >
            <option value="ALL">All Statuses</option>
            <option value="UPLOADED">Uploaded</option>
            <option value="PROCESSING">Processing</option>
            <option value="PROCESSED">Processed</option>
            <option value="PENDING_REVIEW">Pending Review</option>
            <option value="APPROVED">Approved</option>
            <option value="REJECTED">Rejected</option>
            <option value="FAILED">Failed</option>
          </select>
        </div>

        {/* Type filter */}
        <select
          value={typeFilter}
          onChange={(e) => {
            setTypeFilter(e.target.value);
            setCurrentPage(1);
          }}
          className="bg-slate-950/70 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
        >
          <option value="ALL">All Document Types</option>
          {docTypes.map((t) => (
            <option key={t} value={t} className="capitalize">
              {t.replace('_', ' ')}
            </option>
          ))}
        </select>

        {/* Priority filter */}
        <select
          value={priorityFilter}
          onChange={(e) => {
            setPriorityFilter(e.target.value);
            setCurrentPage(1);
          }}
          className="bg-slate-950/70 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
        >
          <option value="ALL">All Review Priorities</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
          <option value="NONE">None</option>
        </select>
      </div>

      {/* Document Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950/60 border-b border-slate-800 text-[11px] uppercase font-semibold text-slate-400">
              <tr>
                <th className="px-5 py-3.5">Document</th>
                <th className="px-4 py-3.5">Classification</th>
                <th className="px-4 py-3.5">Status</th>
                <th className="px-4 py-3.5">Confidence</th>
                <th className="px-4 py-3.5">Review Priority</th>
                <th className="px-4 py-3.5">Created</th>
                <th className="px-5 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {paginatedDocuments.length > 0 ? (
                paginatedDocuments.map((doc) => {
                  const docId = doc.id || doc.document_id || '';
                  const conf = doc.confidence ?? doc.confidence_score;

                  return (
                    <tr 
                      key={docId}
                      className="hover:bg-slate-800/40 transition-colors group"
                    >
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-slate-800 rounded-lg text-slate-400 border border-slate-700">
                            <FileText className="w-4 h-4" />
                          </div>
                          <div className="min-w-0 max-w-xs">
                            <button
                              onClick={() => onSelectDocument(docId)}
                              className="text-slate-200 hover:text-indigo-400 font-semibold truncate block text-left transition-colors"
                            >
                              {doc.filename}
                            </button>
                            <span className="text-[10px] text-slate-500 font-mono">
                              {docId.slice(0, 8)}... · {doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}
                            </span>
                          </div>
                        </div>
                      </td>

                      <td className="px-4 py-4">
                        <span className="capitalize font-medium text-slate-300">
                          {doc.document_type ? doc.document_type.replace('_', ' ') : '—'}
                        </span>
                      </td>

                      <td className="px-4 py-4">
                        <StatusBadge status={doc.status} />
                      </td>

                      <td className="px-4 py-4 font-mono font-medium">
                        {conf !== undefined && conf !== null ? (
                          <div className="flex items-center gap-2">
                            <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  conf >= 0.85
                                    ? 'bg-emerald-500'
                                    : conf >= 0.70
                                    ? 'bg-amber-500'
                                    : 'bg-rose-500'
                                }`}
                                style={{ width: `${conf * 100}%` }}
                              />
                            </div>
                            <span>{(conf * 100).toFixed(0)}%</span>
                          </div>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>

                      <td className="px-4 py-4">
                        {doc.review_priority && doc.review_priority.toString().toUpperCase() !== 'NONE' ? (
                          <ReviewPriorityBadge priority={doc.review_priority} />
                        ) : (
                          <span className="text-[11px] text-slate-500 font-medium">None</span>
                        )}
                      </td>

                      <td className="px-4 py-4 text-slate-400 font-mono text-[11px]">
                        {new Date(doc.created_at || doc.uploaded_at || Date.now()).toLocaleDateString()}
                      </td>

                      <td className="px-5 py-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          {(doc.status === 'UPLOADED' || doc.status === 'uploaded') && (
                            <button
                              onClick={() => onProcessDocument(docId)}
                              className="p-1.5 hover:bg-emerald-500/20 text-emerald-400 rounded-lg transition-colors"
                              title="Process Pipeline"
                            >
                              <Play className="w-3.5 h-3.5" />
                            </button>
                          )}
                          <button
                            onClick={() => onSelectDocument(docId)}
                            className="p-1.5 hover:bg-indigo-500/20 text-indigo-400 rounded-lg transition-colors"
                            title="Open in Workspace"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => {
                              if (window.confirm(`Delete document "${doc.filename}"?`)) {
                                onDeleteDocument(docId);
                              }
                            }}
                            className="p-1.5 hover:bg-rose-500/20 text-rose-400 rounded-lg transition-colors"
                            title="Delete Document"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={7} className="px-5 py-12 text-center text-slate-500">
                    No documents matching the selected filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 bg-slate-950/40 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span>
            Showing <strong className="text-slate-200">{filteredDocuments.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}</strong> to{' '}
            <strong className="text-slate-200">{Math.min(currentPage * pageSize, filteredDocuments.length)}</strong> of{' '}
            <strong className="text-slate-200">{filteredDocuments.length}</strong> documents
          </span>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-lg border border-slate-800 hover:bg-slate-800 disabled:opacity-30 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-2 py-1 font-mono text-slate-300">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-lg border border-slate-800 hover:bg-slate-800 disabled:opacity-30 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
