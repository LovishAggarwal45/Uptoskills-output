import React from 'react';
import { 
  LayoutDashboard, 
  Files, 
  UserCheck, 
  HeartPulse, 
  ShieldCheck
} from 'lucide-react';

export type NavView = 'overview' | 'documents' | 'review' | 'health';

interface SidebarProps {
  currentView: NavView;
  onSelectView: (view: NavView) => void;
  reviewCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  currentView, 
  onSelectView, 
  reviewCount 
}) => {
  const navItems: { id: NavView; label: string; icon: React.ReactNode; badge?: number }[] = [
    {
      id: 'overview',
      label: 'Overview & KPIs',
      icon: <LayoutDashboard className="w-4 h-4" />
    },
    {
      id: 'documents',
      label: 'Document Library',
      icon: <Files className="w-4 h-4" />
    },
    {
      id: 'review',
      label: 'Human Review Queue',
      icon: <UserCheck className="w-4 h-4" />,
      badge: reviewCount > 0 ? reviewCount : undefined
    },
    {
      id: 'health',
      label: 'System Diagnostic',
      icon: <HeartPulse className="w-4 h-4" />
    }
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between p-4 select-none">
      <div className="space-y-6">
        <div className="px-3 pt-2">
          <p className="text-[11px] uppercase tracking-wider font-semibold text-slate-500">Navigation</p>
        </div>

        <nav className="space-y-1.5">
          {navItems.map((item) => {
            const isActive = currentView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectView(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl font-medium text-xs transition-all ${
                  isActive 
                    ? 'bg-indigo-600/15 text-indigo-400 border border-indigo-500/30 shadow-sm' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className={isActive ? 'text-indigo-400' : 'text-slate-500'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </div>

                {item.badge !== undefined && (
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                    isActive ? 'bg-indigo-600 text-white' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                  }`}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Subsystem Integrity Footer */}
      <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Multi-Tier Integrity</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          100% deterministic rules, real OCR extraction, visual overlays & audit trails.
        </p>
        <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-800">
          <span>Phases 1–10 Active</span>
          <span className="text-emerald-400 font-mono">100% Verified</span>
        </div>
      </div>
    </aside>
  );
};
