import React from 'react';
import {
  FileSearch,
  Upload,
  Activity,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Cpu
} from 'lucide-react';
import { SystemHealth } from '../types';

interface NavbarProps {
  health: SystemHealth | null;
  onOpenUpload: () => void;
  onRefresh: () => void;
  isLoading?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  health,
  onOpenUpload,
  onRefresh,
  isLoading = false
}) => {
  const ocrReady = health?.ocr_ready ?? health?.ocr_status?.is_ready ?? false;

  return (
    <header className="h-16 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-6 sticky top-0 z-30 shadow-md">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-blue-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <FileSearch className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-lg text-white tracking-tight">DocuMind AI</span>
            <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              v1.0.0 · Phase 10
            </span>
          </div>
          <p className="text-xs text-slate-400 font-medium">Enterprise Document Intelligence Platform</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* System Health Quick Status */}
        <div className="hidden md:flex items-center gap-3 bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5 text-xs text-slate-300">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Tesseract:</span>
            {ocrReady ? (
              <span className="flex items-center gap-1 text-emerald-400 font-medium">
                <CheckCircle2 className="w-3 h-3" /> Ready
              </span>
            ) : (
              <span className="flex items-center gap-1 text-amber-400 font-medium">
                <AlertTriangle className="w-3 h-3" /> Degraded
              </span>
            )}
          </div>
          <div className="w-px h-3.5 bg-slate-700" />
          <div className="flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-blue-400" />
            <span>DB:</span>
            <span className="text-emerald-400 font-medium">SQLite WAL</span>
          </div>
        </div>

        {/* Global Action Buttons */}
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors border border-transparent hover:border-slate-700 disabled:opacity-50"
          title="Refresh Data"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-indigo-400' : ''}`} />
        </button>

        <button
          onClick={onOpenUpload}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs px-3.5 py-2 rounded-lg shadow-lg shadow-indigo-600/25 transition-all border border-indigo-500 hover:scale-[1.02] active:scale-[0.98]"
        >
          <Upload className="w-4 h-4" />
          <span>Upload Document</span>
        </button>
      </div>
    </header>
  );
};
