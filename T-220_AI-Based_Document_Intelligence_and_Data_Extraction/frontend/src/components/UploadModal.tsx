import React, { useState, useRef } from 'react';
import { api } from '../api/client';
import { Upload, X, FileUp, AlertCircle, FileText, ArrowRight } from 'lucide-react';

interface Props {
  isOpen?: boolean;
  onClose: () => void;
  onUploadSuccess?: (uploadedDocId?: string) => Promise<void> | void;
  onUploaded?: (uploadedDocId?: string) => void;
}

export const UploadModal: React.FC<Props> = ({ isOpen = true, onClose, onUploadSuccess, onUploaded }) => {
  const [dragOver, setDragOver] = useState<boolean>(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [autoProcess, setAutoProcess] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    const valid: File[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.size > 50 * 1024 * 1024) {
        setError(`File '${file.name}' exceeds maximum size of 50 MB.`);
        return;
      }
      valid.push(file);
    }
    setSelectedFiles((prev) => [...prev, ...valid]);
    setError(null);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUploadSubmit = async () => {
    if (selectedFiles.length === 0) return;
    setUploading(true);
    setError(null);

    try {
      let lastDocId: string | undefined;
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        setUploadProgress(`Processing ${i + 1} of ${selectedFiles.length}: ${file.name}...`);
        const res = await api.uploadDocument(file, autoProcess);
        lastDocId = res.document_id;
      }
      if (onUploadSuccess) {
        await onUploadSuccess(lastDocId);
      } else if (onUploaded) {
        onUploaded(lastDocId);
      }
      onClose();
    } catch (err: any) {
      setError(err.message || 'Upload failed.');
    } finally {
      setUploading(false);
      setUploadProgress('');
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl animate-in fade-in zoom-in duration-200">
        <div className="flex items-center justify-between px-6 py-4 bg-slate-800/80 border-b border-slate-700">
          <div className="flex items-center gap-2.5">
            <Upload className="w-5 h-5 text-indigo-400" />
            <h3 className="font-semibold text-slate-100 text-base">Upload Document</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {error && (
            <div className="p-3 bg-rose-950/80 border border-rose-700 rounded-lg text-rose-300 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Drag & Drop Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
              dragOver
                ? 'border-indigo-400 bg-indigo-950/30'
                : 'border-slate-700 hover:border-slate-600 bg-slate-950/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp"
              onChange={(e) => handleFiles(e.target.files)}
              className="hidden"
            />
            <FileUp className="w-10 h-10 text-slate-500 mx-auto mb-3" />
            <p className="text-sm font-medium text-slate-200">
              Click to select or drag and drop document files
            </p>
            <p className="text-xs text-slate-500 mt-1 font-mono">
              Supported: PDF, PNG, JPG, JPEG, TIFF, BMP (Max 50 MB)
            </p>
          </div>

          {/* Selected files list */}
          {selectedFiles.length > 0 && (
            <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Selected Documents ({selectedFiles.length})
              </div>
              {selectedFiles.map((f, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-2 rounded-lg bg-slate-950 border border-slate-800 text-xs"
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
                    <span className="truncate text-slate-200">{f.name}</span>
                    <span className="text-slate-500 font-mono text-[10px]">
                      ({(f.size / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(i);
                    }}
                    className="p-1 text-slate-500 hover:text-rose-400 transition"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Options */}
          <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
            <label className="flex items-center gap-2 cursor-pointer select-none text-xs text-slate-300">
              <input
                type="checkbox"
                checked={autoProcess}
                onChange={(e) => setAutoProcess(e.target.checked)}
                className="rounded bg-slate-800 border-slate-700 text-indigo-500 focus:ring-0"
              />
              <span>Execute full pipeline (OCR, Extraction, Validation, Explainability) immediately</span>
            </label>
          </div>

          {uploading && (
            <div className="flex items-center gap-2.5 p-3 bg-indigo-950/60 border border-indigo-800 rounded-lg text-indigo-300 text-xs animate-pulse">
              <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin shrink-0" />
              <span>{uploadProgress || 'Uploading and running pipeline...'}</span>
            </div>
          )}

          <div className="pt-3 border-t border-slate-800 flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={onClose}
              disabled={uploading}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleUploadSubmit}
              disabled={selectedFiles.length === 0 || uploading}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-lg flex items-center gap-1.5 transition disabled:opacity-40"
            >
              <span>{autoProcess ? 'Upload & Process' : 'Upload Only'}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
