import React, { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar, NavView } from './components/Sidebar';
import { OverviewView } from './views/OverviewView';
import { DocumentsView } from './views/DocumentsView';
import { ReviewQueueView } from './views/ReviewQueueView';
import { WorkspaceView } from './views/WorkspaceView';
import { HealthView } from './views/HealthView';
import { UploadModal } from './components/UploadModal';
import { api } from './api/client';
import { 
  SystemHealth, 
  SystemStats, 
  DocumentItem, 
  DocumentDetail,
  FieldCorrectionPayload,
  ReviewDecisionPayload 
} from './types';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  // Navigation
  const [currentView, setCurrentView] = useState<NavView>('overview');
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  // Data state
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocDetail, setSelectedDocDetail] = useState<DocumentDetail | null>(null);

  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [isWorkspaceLoading, setIsWorkspaceLoading] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load global system state
  const loadSystemData = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const [healthData, statsData, docsData] = await Promise.all([
        api.getHealth(),
        api.getStatsOverview(),
        api.listDocuments({ limit: 100 })
      ]);
      setHealth(healthData);
      setStats(statsData);
      setDocuments(docsData.documents);
    } catch (err: any) {
      console.error('Failed to load system data:', err);
      setErrorMessage(err.message || 'Failed to connect to DocuMind API server');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSystemData();
  }, [loadSystemData]);

  // Load selected document details
  const loadDocumentDetail = useCallback(async (docId: string) => {
    setIsWorkspaceLoading(true);
    try {
      const detail = await api.getDocument(docId);
      setSelectedDocDetail(detail);
    } catch (err: any) {
      console.error('Failed to load document details:', err);
      alert(`Error loading document: ${err.message}`);
    } finally {
      setIsWorkspaceLoading(false);
    }
  }, []);

  // Handle document selection
  const handleSelectDocument = (docId: string) => {
    setSelectedDocId(docId);
    loadDocumentDetail(docId);
  };

  // Handle document processing
  const handleProcessDocument = async (docId: string) => {
    setIsLoading(true);
    try {
      await api.processDocument(docId);
      await loadSystemData();
      if (selectedDocId === docId) {
        await loadDocumentDetail(docId);
      }
    } catch (err: any) {
      console.error('Failed to process document:', err);
      alert(`Processing error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle document deletion
  const handleDeleteDocument = async (docId: string) => {
    setIsLoading(true);
    try {
      await api.deleteDocument(docId);
      if (selectedDocId === docId) {
        setSelectedDocId(null);
        setSelectedDocDetail(null);
      }
      await loadSystemData();
    } catch (err: any) {
      console.error('Failed to delete document:', err);
      alert(`Delete error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle correction save
  const handleSaveCorrection = async (docId: string, payload: FieldCorrectionPayload) => {
    try {
      await api.saveCorrection(docId, payload);
      await loadDocumentDetail(docId);
      await loadSystemData();
    } catch (err: any) {
      console.error('Failed to save correction:', err);
      alert(`Correction error: ${err.message}`);
    }
  };

  // Handle review decision
  const handleSubmitDecision = async (docId: string, payload: ReviewDecisionPayload) => {
    try {
      await api.submitReviewDecision(docId, payload);
      await loadDocumentDetail(docId);
      await loadSystemData();
    } catch (err: any) {
      console.error('Failed to submit review decision:', err);
      alert(`Review decision error: ${err.message}`);
    }
  };

  const pendingReviewCount = documents.filter(
    (d) => d.status === 'PENDING_REVIEW' || d.status === 'needs_review' || (d.review_priority && d.review_priority.toString().toUpperCase() !== 'NONE')
  ).length;

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Top Navigation */}
      <Navbar
        health={health}
        onOpenUpload={() => setIsUploadOpen(true)}
        onRefresh={loadSystemData}
        isLoading={isLoading}
      />

      {/* Global Error Banner */}
      {errorMessage && (
        <div className="bg-rose-500/10 border-b border-rose-500/20 px-6 py-2.5 flex items-center justify-between text-xs text-rose-300">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-rose-400 hover:text-white text-xs font-semibold"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Workspace View replaces standard layout when active */}
        {selectedDocId && selectedDocDetail ? (
          <div className="flex-1 h-full min-h-0">
            <WorkspaceView
              document={selectedDocDetail}
              isLoading={isWorkspaceLoading}
              onBack={() => {
                setSelectedDocId(null);
                setSelectedDocDetail(null);
              }}
              onProcess={handleProcessDocument}
              onSaveCorrection={handleSaveCorrection}
              onSubmitDecision={handleSubmitDecision}
              onRefresh={() => loadDocumentDetail(selectedDocId)}
            />
          </div>
        ) : (
          <>
            {/* Sidebar Navigation */}
            <Sidebar
              currentView={currentView}
              onSelectView={(view) => {
                setCurrentView(view);
                setSelectedDocId(null);
              }}
              reviewCount={pendingReviewCount}
            />

            {/* View Switching */}
            <main className="flex-1 bg-slate-950 min-h-0 overflow-y-auto">
              {currentView === 'overview' && (
                <OverviewView
                  stats={stats}
                  documents={documents}
                  onSelectDocument={handleSelectDocument}
                  onNavigateToReview={() => setCurrentView('review')}
                  onNavigateToDocuments={() => setCurrentView('documents')}
                />
              )}

              {currentView === 'documents' && (
                <DocumentsView
                  documents={documents}
                  isLoading={isLoading}
                  onSelectDocument={handleSelectDocument}
                  onProcessDocument={handleProcessDocument}
                  onDeleteDocument={handleDeleteDocument}
                  onOpenUpload={() => setIsUploadOpen(true)}
                  onRefresh={loadSystemData}
                />
              )}

              {currentView === 'review' && (
                <ReviewQueueView
                  documents={documents}
                  isLoading={isLoading}
                  onSelectDocument={handleSelectDocument}
                  onRefresh={loadSystemData}
                />
              )}

              {currentView === 'health' && (
                <HealthView
                  health={health}
                  isLoading={isLoading}
                  onRefresh={loadSystemData}
                />
              )}
            </main>
          </>
        )}
      </div>

      {/* Document Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={async (newDocId) => {
          await loadSystemData();
          if (newDocId) {
            handleSelectDocument(newDocId);
          }
        }}
      />
    </div>
  );
};
