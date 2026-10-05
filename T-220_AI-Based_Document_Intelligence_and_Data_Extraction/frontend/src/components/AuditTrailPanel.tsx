import React from 'react';
import { AuditRecord, HumanCorrection, ReviewDecision } from '../types';
import { Clock, User, Cpu, ShieldCheck, Edit3 } from 'lucide-react';

interface Props {
  auditTrail: AuditRecord[];
  corrections?: HumanCorrection[];
  decisions?: ReviewDecision[];
}

export const AuditTrailPanel: React.FC<Props> = ({ auditTrail }) => {
  if (!auditTrail || auditTrail.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-slate-500 text-center">
        <Clock className="w-10 h-10 text-slate-700 mb-2" />
        <span className="text-xs">No audit events recorded yet.</span>
      </div>
    );
  }

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'FIELD_CORRECTED':
      case 'CORRECTION_ADDED':
        return <Edit3 className="w-4 h-4 text-amber-400" />;
      case 'REVIEW_DECISION_RECORDED':
      case 'DECISION_APPROVED':
        return <ShieldCheck className="w-4 h-4 text-emerald-400" />;
      case 'PIPELINE_COMPLETED':
        return <ShieldCheck className="w-4 h-4 text-indigo-400" />;
      default:
        return <Cpu className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <h3 className="font-semibold text-slate-200 text-xs uppercase tracking-wider flex items-center gap-2">
          <Clock className="w-4 h-4 text-indigo-400" />
          <span>Immutable Document Audit Trail</span>
        </h3>
        <span className="text-[11px] font-mono text-slate-500">{auditTrail.length} records</span>
      </div>

      <div className="relative pl-6 space-y-4 before:content-[''] before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
        {auditTrail.map((event, idx) => {
          const isHuman = event.actor !== 'system';

          return (
            <div key={event.audit_id || idx} className="relative group">
              {/* Dot marker */}
              <div className="absolute -left-6 mt-1 w-4 h-4 rounded-full bg-slate-900 border-2 border-slate-700 flex items-center justify-center group-hover:border-indigo-500 transition">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3 shadow-sm hover:border-slate-700 transition">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2">
                    {getEventIcon(event.event_type)}
                    <span className="font-mono text-xs font-semibold text-slate-200">
                      {event.event_type}
                    </span>
                  </div>

                  <span className="font-mono text-[10px] text-slate-500">
                    {new Date(event.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                    })}
                  </span>
                </div>

                <div className="text-xs text-slate-300 mb-2">{event.description}</div>

                <div className="flex items-center gap-2 text-[10px] text-slate-500 border-t border-slate-800/60 pt-1.5">
                  <span className="flex items-center gap-1">
                    {isHuman ? <User className="w-3 h-3 text-amber-400" /> : <Cpu className="w-3 h-3" />}
                    <span>Actor: <strong className="text-slate-400">{event.actor}</strong></span>
                  </span>
                  <span>•</span>
                  <span>{new Date(event.timestamp).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
