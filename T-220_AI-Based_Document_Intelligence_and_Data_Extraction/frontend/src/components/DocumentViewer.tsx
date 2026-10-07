import React, { useState, useRef, useEffect } from 'react';
import { BoundingBox } from '../types';
import {
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  Layers,
  Eye,
  EyeOff,
  AlertCircle,
  RotateCcw
} from 'lucide-react';

interface Props {
  documentId: string;
  pageCount?: number;
  totalPages?: number;
  currentPage: number;
  onPageChange: (page: number) => void;
  highlightBox?: BoundingBox | null;
  activeTargetBox?: BoundingBox | null;
  targetLabel?: string | null;
  visualizations?: any;
  hasOverlays?: boolean;
  onClearHighlight?: () => void;
}

export const DocumentViewer: React.FC<Props> = ({
  documentId,
  pageCount,
  totalPages: propTotalPages,
  currentPage,
  onPageChange,
  highlightBox,
  activeTargetBox: propTargetBox,
  targetLabel,
  hasOverlays = true,
}) => {
  const totalPages = pageCount || propTotalPages || 1;
  const targetBox = highlightBox || propTargetBox;

  const [zoom, setZoom] = useState<number>(1.0);
  const [useOverlay, setUseOverlay] = useState<boolean>(true);
  const [layerFilters, setLayerFilters] = useState({
    ocr: true,
    fields: true,
    tables: true,
    validation: true,
    review: true,
  });
  const [showLayerMenu, setShowLayerMenu] = useState<boolean>(false);
  const [imageLoaded, setImageLoaded] = useState<boolean>(false);
  const [imageError, setImageError] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const imageUrl = useOverlay && hasOverlays
    ? `/api/documents/${documentId}/overlay-image/${currentPage}`
    : `/api/documents/${documentId}/page-image/${currentPage}`;

  useEffect(() => {
    if (!hasOverlays) {
      setUseOverlay(false);
    }
  }, [hasOverlays]);

  useEffect(() => {
    setImageLoaded(false);
    setImageError(false);
  }, [currentPage, useOverlay, documentId]);

  const handleZoomIn = () => setZoom((prev) => Math.min(3.0, prev + 0.2));
  const handleZoomOut = () => setZoom((prev) => Math.max(0.4, prev - 0.2));
  const handleZoomReset = () => setZoom(1.0);

  const handleImageLoad = () => {
    setImageLoaded(true);
    setImageError(false);
  };

  const handleImageError = () => {
    if (useOverlay) {
      // Seamlessly fall back to the original rendered page image
      setUseOverlay(false);
    } else {
      setImageError(true);
      setImageLoaded(false);
    }
  };

  // Convert target bounding box to scaled pixel offsets
  const renderTargetHighlight = () => {
    if (!targetBox || !imageLoaded) return null;

    const { xmin, ymin, xmax, ymax } = targetBox;
    const width = Math.max(8, xmax - xmin);
    const height = Math.max(8, ymax - ymin);

    return (
      <div
        className="absolute border-2 border-rose-500 bg-rose-500/20 rounded shadow-lg pointer-events-none transition-all duration-300 animate-pulse z-30"
        style={{
          left: `${xmin}px`,
          top: `${ymin}px`,
          width: `${width}px`,
          height: `${height}px`,
        }}
      >
        {targetLabel && (
          <div className="absolute -top-7 left-0 bg-rose-600 text-white font-mono text-xs px-2 py-0.5 rounded shadow whitespace-nowrap z-40">
            {targetLabel}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl relative">
      {/* Top Toolbar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900 border-b border-slate-800 z-20 shrink-0">
        {/* Page navigation */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => onPageChange(Math.max(1, currentPage - 1))}
            disabled={currentPage <= 1}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300 transition-colors"
            title="Previous Page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs font-mono font-medium text-slate-300 px-2 py-1 bg-slate-800/80 rounded border border-slate-700">
            Page {currentPage} of {Math.max(1, totalPages)}
          </span>
          <button
            onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage >= totalPages}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300 transition-colors"
            title="Next Page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-slate-800/80 p-0.5 rounded-lg border border-slate-700">
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded text-slate-300 hover:text-white hover:bg-slate-700 transition"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleZoomReset}
            className="px-2 py-1 text-xs font-mono text-slate-300 hover:text-white transition flex items-center gap-1"
            title="Reset Zoom"
          >
            <RotateCcw className="w-3 h-3 text-slate-400" />
            <span>{Math.round(zoom * 100)}%</span>
          </button>
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded text-slate-300 hover:text-white hover:bg-slate-700 transition"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Overlay / Original Mode Switch */}
        <div className="flex items-center gap-2">
          {hasOverlays && (
            <button
              onClick={() => setUseOverlay(!useOverlay)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                useOverlay
                  ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300 hover:bg-indigo-600/40'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {useOverlay ? <Eye className="w-3.5 h-3.5 text-indigo-400" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span>{useOverlay ? 'Evidence Overlay' : 'Original Image'}</span>
            </button>
          )}

          {/* Layer Controls Dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowLayerMenu(!showLayerMenu)}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              title="Toggle Evidence Layers"
            >
              <Layers className="w-4 h-4" />
            </button>

            {showLayerMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-slate-900 border border-slate-700 rounded-lg p-2 shadow-2xl z-50 text-xs space-y-1.5">
                <div className="font-semibold text-slate-400 px-2 py-1 uppercase tracking-wider text-[10px] border-b border-slate-800">
                  Visual Layers
                </div>
                {Object.entries({
                  fields: '[FLD] Extracted Fields',
                  tables: '[TBL] Tables & Grid',
                  validation: '[VAL] Validation Issues',
                  review: '[REV] Review Targets',
                  ocr: '[OCR] Word Tokens',
                }).map(([k, label]) => (
                  <label
                    key={k}
                    className="flex items-center gap-2 px-2 py-1 rounded hover:bg-slate-800 cursor-pointer text-slate-300 select-none"
                  >
                    <input
                      type="checkbox"
                      checked={(layerFilters as any)[k]}
                      onChange={(e) =>
                        setLayerFilters({ ...layerFilters, [k]: e.target.checked })
                      }
                      className="rounded bg-slate-800 border-slate-700 text-indigo-500 focus:ring-0"
                    />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Canvas Scroll Area */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto p-4 flex items-center justify-center bg-slate-950/80 relative"
      >
        {!imageLoaded && !imageError && (
          <div className="flex flex-col items-center justify-center gap-3 text-slate-400 py-20">
            <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs">Loading high-resolution page rendering...</span>
          </div>
        )}

        {imageError && (
          <div className="flex flex-col items-center justify-center gap-3 text-slate-400 py-20 max-w-sm text-center">
            <AlertCircle className="w-10 h-10 text-amber-400" />
            <span className="text-sm font-semibold text-slate-200">Page image not yet generated</span>
            <span className="text-xs text-slate-400">
              Run pipeline processing on this document to produce rendered page and visual explainability overlays.
            </span>
          </div>
        )}

        <div
          className="relative transition-transform duration-150 origin-center shadow-2xl rounded"
          style={{
            transform: `scale(${zoom})`,
            display: imageLoaded ? 'block' : 'none',
          }}
        >
          <img
            ref={imgRef}
            src={imageUrl}
            alt={`Document Page ${currentPage}`}
            onLoad={handleImageLoad}
            onError={handleImageError}
            className="max-w-none rounded border border-slate-800 block pointer-events-auto"
            style={{ maxHeight: 'calc(100vh - 220px)' }}
          />
          {renderTargetHighlight()}
        </div>
      </div>
    </div>
  );
};
