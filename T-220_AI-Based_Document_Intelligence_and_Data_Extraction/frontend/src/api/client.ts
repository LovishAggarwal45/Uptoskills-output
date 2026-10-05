import {
  AuditRecord,
  DocumentRecord,
  HealthStatus,
  HumanCorrection,
  OverviewStats,
} from '../types';

const API_BASE = '/api';

export class ApiError extends Error {
  constructor(public status: number, message: string, public details?: any) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) {
    let errorMsg = `HTTP Error ${res.status}`;
    let details: any = null;
    try {
      const errData = await res.json();
      errorMsg = errData.detail || errData.message || errorMsg;
      details = errData;
    } catch {
      // Ignore text parse errors
    }
    throw new ApiError(res.status, errorMsg, details);
  }
  return res.json();
}

function normalizeDoc(doc: any): DocumentRecord {
  if (!doc) return doc;
  const id = doc.document_id || doc.id || '';
  return {
    ...doc,
    id,
    document_id: id,
  };
}

export const api = {
  // Health & Diagnostics
  getHealth: (): Promise<HealthStatus> => fetchJson<HealthStatus>('/health'),

  // Overview Stats
  getStats: (): Promise<OverviewStats> => fetchJson<OverviewStats>('/stats/overview'),
  getStatsOverview: (): Promise<OverviewStats> => fetchJson<OverviewStats>('/stats/overview'),

  // Documents
  listDocuments: async (params?: {
    status?: string;
    type?: string;
    priority?: string;
    validation?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ total: number; limit: number; offset: number; items: DocumentRecord[]; documents: DocumentRecord[] }> => {
    const query = new URLSearchParams();
    if (params?.status) query.append('status', params.status);
    if (params?.type) query.append('type', params.type);
    if (params?.priority) query.append('priority', params.priority);
    if (params?.validation) query.append('validation', params.validation);
    if (params?.search) query.append('search', params.search);
    if (params?.limit) query.append('limit', params.limit.toString());
    if (params?.offset) query.append('offset', params.offset.toString());
    const qs = query.toString();
    const data = await fetchJson<{ total: number; limit: number; offset: number; items: DocumentRecord[] }>(
      `/documents${qs ? `?${qs}` : ''}`
    );
    const normalizedItems = (data.items || []).map(normalizeDoc);
    return {
      ...data,
      items: normalizedItems,
      documents: normalizedItems,
    };
  },

  getDocument: async (id: string): Promise<DocumentRecord> => {
    if (!id || id === 'undefined' || id === 'null') {
      throw new ApiError(400, 'Invalid document ID');
    }
    const doc = await fetchJson<DocumentRecord>(`/documents/${id}`);
    return normalizeDoc(doc);
  },

  uploadDocument: async (file: File, autoProcess = false): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('auto_process', autoProcess ? 'true' : 'false');

    const res = await fetch(`${API_BASE}/documents`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new ApiError(res.status, err.detail || 'Upload failed', err);
    }
    const data = await res.json();
    return normalizeDoc(data);
  },

  processDocument: (id: string): Promise<any> =>
    fetchJson<any>(`/documents/${id}/process`, { method: 'POST' }),

  getDocumentResults: (id: string): Promise<any> =>
    fetchJson<any>(`/documents/${id}/results`),

  getDocumentVisualizations: (id: string): Promise<any> =>
    fetchJson<any>(`/documents/${id}/visualizations`),

  deleteDocument: (id: string): Promise<any> =>
    fetchJson<any>(`/documents/${id}`, { method: 'DELETE' }),

  // Review Queue
  getReviewQueue: async (params?: {
    priority?: string;
    type?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ total_queued: number; limit: number; offset: number; items: DocumentRecord[] }> => {
    const query = new URLSearchParams();
    if (params?.priority) query.append('priority', params.priority);
    if (params?.type) query.append('type', params.type);
    if (params?.limit) query.append('limit', params.limit.toString());
    if (params?.offset) query.append('offset', params.offset.toString());
    const qs = query.toString();
    const data = await fetchJson<{ total_queued: number; limit: number; offset: number; items: DocumentRecord[] }>(
      `/review-queue${qs ? `?${qs}` : ''}`
    );
    return {
      ...data,
      items: (data.items || []).map(normalizeDoc),
    };
  },

  // Corrections & Decisions
  getCorrections: (documentId: string): Promise<HumanCorrection[]> =>
    fetchJson<HumanCorrection[]>(`/documents/${documentId}/corrections`),

  addCorrection: (
    documentId: string,
    data: {
      field_name: string;
      original_value?: string;
      corrected_value: string;
      reviewer_id?: string;
      reason?: string;
    }
  ): Promise<HumanCorrection> =>
    fetchJson<HumanCorrection>(`/documents/${documentId}/corrections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  saveCorrection: (
    documentId: string,
    data: {
      field_name: string;
      original_value?: string;
      corrected_value: string;
      reviewer_id?: string;
      reason?: string;
    }
  ): Promise<HumanCorrection> =>
    fetchJson<HumanCorrection>(`/documents/${documentId}/corrections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  submitReviewDecision: (
    documentId: string,
    data: {
      action: string;
      reviewer_id?: string;
      notes?: string;
    }
  ): Promise<any> =>
    fetchJson<any>(`/documents/${documentId}/review-decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  getAuditTrail: (documentId: string): Promise<AuditRecord[]> =>
    fetchJson<AuditRecord[]>(`/documents/${documentId}/audit-trail`),

  // Image helpers
  getPageImageUrl: (documentId: string, pageNumber: number): string =>
    `${API_BASE}/documents/${documentId}/page-image/${pageNumber}`,

  getOverlayImageUrl: (documentId: string, pageNumber: number): string =>
    `${API_BASE}/documents/${documentId}/overlay-image/${pageNumber}`,

  // Export URLs
  getExportUrl: (documentId: string, format: 'json' | 'reviewed' | 'csv' | 'report'): string =>
    `${API_BASE}/documents/${documentId}/export/${format}`,
};
