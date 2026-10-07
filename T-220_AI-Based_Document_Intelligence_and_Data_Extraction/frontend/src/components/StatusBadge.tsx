import React from 'react';
import {
  ConfidenceBand,
  DocumentStatus,
  DocumentType,
  ReviewPriority,
  ValidationSeverity,
  ValidationStatus,
} from '../types';
import {
  CheckCircle2,
  Clock,
  AlertTriangle,
  AlertOctagon,
  XCircle,
  FileText,
  Receipt,
  FileSpreadsheet,
  FileQuestion,
  HelpCircle,
} from 'lucide-react';

interface Props {
  type?: 'status' | 'doc_type' | 'confidence' | 'priority' | 'severity' | 'validation';
  value?: any;
  status?: any;
  priority?: any;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const StatusBadge: React.FC<Props> = ({
  type = 'status',
  value,
  status,
  size = 'md',
  showIcon = true,
}) => {
  const effectiveValue = status !== undefined ? status : value;
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3 py-1.5',
  }[size];

  if (type === 'status') {
    const stat = (effectiveValue || 'uploaded').toString().toLowerCase() as DocumentStatus;
    const configs: Record<string, { label: string; badge: string; bg: string; text: string; icon: any }> = {
      uploaded: { label: 'Uploaded', badge: '[UPL]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-300', icon: Clock },
      processing: { label: 'Processing', badge: '[PROC]', bg: 'bg-blue-950/80 border-blue-800 animate-pulse', text: 'text-blue-300', icon: Clock },
      processed: { label: 'Processed', badge: '[PROC]', bg: 'bg-emerald-950/80 border-emerald-700', text: 'text-emerald-300', icon: CheckCircle2 },
      completed: { label: 'Completed', badge: '[PASS]', bg: 'bg-emerald-950/80 border-emerald-700', text: 'text-emerald-300', icon: CheckCircle2 },
      approved: { label: 'Approved', badge: '[APPR]', bg: 'bg-emerald-950/80 border-emerald-700', text: 'text-emerald-300', icon: CheckCircle2 },
      pending_review: { label: 'Pending Review', badge: '[REVIEW]', bg: 'bg-amber-950/80 border-amber-700', text: 'text-amber-300', icon: AlertTriangle },
      needs_review: { label: 'Needs Review', badge: '[REVIEW]', bg: 'bg-amber-950/80 border-amber-700', text: 'text-amber-300', icon: AlertTriangle },
      rejected: { label: 'Rejected', badge: '[REJ]', bg: 'bg-rose-950/80 border-rose-800', text: 'text-rose-300', icon: XCircle },
      failed: { label: 'Failed', badge: '[FAIL]', bg: 'bg-rose-950/80 border-rose-800', text: 'text-rose-300', icon: XCircle },
    };
    const c = configs[stat] || configs.uploaded;
    const Icon = c.icon;
    return (
      <span className={`inline-flex items-center gap-1.5 font-medium rounded-md border ${c.bg} ${c.text} ${sizeClasses}`}>
        {showIcon && <Icon className="w-3.5 h-3.5" />}
        <span className="font-mono opacity-75">{c.badge}</span>
        <span>{c.label}</span>
      </span>
    );
  }

  if (type === 'doc_type') {
    const dt = (effectiveValue || 'unknown').toString().toLowerCase() as DocumentType;
    const configs: Record<string, { label: string; badge: string; bg: string; text: string; icon: any }> = {
      invoice: { label: 'Tax Invoice', badge: '[INV]', bg: 'bg-indigo-950/80 border-indigo-700', text: 'text-indigo-300', icon: FileText },
      receipt: { label: 'POS Receipt', badge: '[RCPT]', bg: 'bg-teal-950/80 border-teal-700', text: 'text-teal-300', icon: Receipt },
      form: { label: 'Intake Form', badge: '[FORM]', bg: 'bg-violet-950/80 border-violet-700', text: 'text-violet-300', icon: FileSpreadsheet },
      general_document: { label: 'General Doc', badge: '[DOC]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-300', icon: FileText },
      unknown: { label: 'Unclassified', badge: '[UNK]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-400', icon: FileQuestion },
    };
    const c = configs[dt] || configs.unknown;
    const Icon = c.icon;
    return (
      <span className={`inline-flex items-center gap-1.5 font-medium rounded-md border ${c.bg} ${c.text} ${sizeClasses}`}>
        {showIcon && <Icon className="w-3.5 h-3.5" />}
        <span className="font-mono opacity-75">{c.badge}</span>
        <span>{c.label}</span>
      </span>
    );
  }

  if (type === 'confidence') {
    const band = (effectiveValue || 'low').toString().toLowerCase() as ConfidenceBand;
    const configs: Record<string, { label: string; badge: string; bg: string; text: string }> = {
      high: { label: 'High (≥85%)', badge: '[HIGH]', bg: 'bg-emerald-950/80 border-emerald-700', text: 'text-emerald-300' },
      medium: { label: 'Medium (65-85%)', badge: '[MED]', bg: 'bg-blue-950/80 border-blue-700', text: 'text-blue-300' },
      low: { label: 'Low (40-65%)', badge: '[LOW]', bg: 'bg-amber-950/80 border-amber-700', text: 'text-amber-300' },
      very_low: { label: 'Very Low (<40%)', badge: '[VERY_LOW]', bg: 'bg-rose-950/80 border-rose-800', text: 'text-rose-300' },
    };
    const c = configs[band] || configs.low;
    return (
      <span className={`inline-flex items-center gap-1 font-medium rounded-md border ${c.bg} ${c.text} ${sizeClasses}`}>
        <span className="font-mono font-semibold">{c.badge}</span>
        <span>{c.label}</span>
      </span>
    );
  }

  if (type === 'priority') {
    const prio = (effectiveValue || 'low').toString().toLowerCase() as ReviewPriority;
    const configs: Record<string, { label: string; badge: string; bg: string; text: string; icon: any }> = {
      urgent: { label: 'Urgent Priority', badge: '[URGENT]', bg: 'bg-rose-950 border-rose-600', text: 'text-rose-300', icon: AlertOctagon },
      critical: { label: 'Critical Priority', badge: '[CRIT]', bg: 'bg-rose-950 border-rose-600', text: 'text-rose-300', icon: AlertOctagon },
      high: { label: 'High Priority', badge: '[HIGH]', bg: 'bg-amber-950 border-amber-600', text: 'text-amber-300', icon: AlertTriangle },
      medium: { label: 'Medium Priority', badge: '[MED]', bg: 'bg-yellow-950 border-yellow-700', text: 'text-yellow-300', icon: Clock },
      low: { label: 'Low Priority', badge: '[LOW]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-300', icon: CheckCircle2 },
      none: { label: 'No Review Needed', badge: '[NONE]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-400', icon: CheckCircle2 },
    };
    const c = configs[prio] || configs.low;
    const Icon = c.icon;
    return (
      <span className={`inline-flex items-center gap-1.5 font-medium rounded-md border ${c.bg} ${c.text} ${sizeClasses}`}>
        {showIcon && <Icon className="w-3.5 h-3.5" />}
        <span className="font-mono font-bold">{c.badge}</span>
        <span>{c.label}</span>
      </span>
    );
  }

  if (type === 'severity') {
    const sev = (effectiveValue || 'info').toString().toLowerCase() as ValidationSeverity;
    const configs: Record<string, { label: string; badge: string; bg: string; text: string; icon: any }> = {
      critical: { label: 'Critical Violation', badge: '[CRITICAL]', bg: 'bg-rose-950 border-rose-600', text: 'text-rose-300', icon: AlertOctagon },
      error: { label: 'Validation Error', badge: '[ERR]', bg: 'bg-rose-950/80 border-rose-700', text: 'text-rose-300', icon: XCircle },
      warning: { label: 'Soft Warning', badge: '[WARN]', bg: 'bg-amber-950/80 border-amber-700', text: 'text-amber-300', icon: AlertTriangle },
      info: { label: 'Info Note', badge: '[INFO]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-300', icon: HelpCircle },
    };
    const c = configs[sev] || configs.info;
    const Icon = c.icon;
    return (
      <span className={`inline-flex items-center gap-1.5 font-medium rounded-md border ${c.bg} ${c.text} ${sizeClasses}`}>
        {showIcon && <Icon className="w-3.5 h-3.5" />}
        <span className="font-mono font-bold">{c.badge}</span>
        <span>{c.label}</span>
      </span>
    );
  }

  // validation status
  const stat = (effectiveValue || 'unknown').toString().toLowerCase() as ValidationStatus;
  const configs: Record<string, { label: string; badge: string; bg: string; text: string; icon: any }> = {
    valid: { label: 'Valid', badge: '[VALID]', bg: 'bg-emerald-950/80 border-emerald-700', text: 'text-emerald-300', icon: CheckCircle2 },
    warning: { label: 'Warning', badge: '[WARN]', bg: 'bg-amber-950/80 border-amber-700', text: 'text-amber-300', icon: AlertTriangle },
    invalid: { label: 'Invalid', badge: '[INVALID]', bg: 'bg-rose-950/80 border-rose-800', text: 'text-rose-300', icon: XCircle },
    unknown: { label: 'Unverified', badge: '[UNCHECKED]', bg: 'bg-slate-800 border-slate-700', text: 'text-slate-400', icon: HelpCircle },
  };
  const c = configs[stat] || configs.unknown;
  const Icon = c.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 font-medium rounded-md border ${c.bg} ${c.text} ${sizeClasses}`}>
      {showIcon && <Icon className="w-3.5 h-3.5" />}
      <span className="font-mono opacity-80">{c.badge}</span>
      <span>{c.label}</span>
    </span>
  );
};

export const ReviewPriorityBadge: React.FC<{ priority?: any; size?: 'sm' | 'md' | 'lg' }> = ({
  priority,
  size = 'md',
}) => {
  return <StatusBadge type="priority" value={priority} size={size} />;
};

export const ConfidenceBadge: React.FC<{ band?: any; size?: 'sm' | 'md' | 'lg' }> = ({
  band,
  size = 'md',
}) => {
  return <StatusBadge type="confidence" value={band} size={size} />;
};
