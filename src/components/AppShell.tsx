import { useState } from 'react';
import {
  LayoutDashboard,
  Search,
  Upload,
  GitCompare,
  Lightbulb,
  Menu,
  X,
  ShoppingBag,
  FlaskConical,
} from 'lucide-react';
import type { PageId } from '@/types/navigation';

export const NAV_ITEMS: { id: PageId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'single', label: 'Single Review Analysis', icon: Search },
  { id: 'bulk', label: 'Bulk Review Analysis', icon: Upload },
  { id: 'models', label: 'Model Comparison', icon: GitCompare },
  { id: 'insights', label: 'Customer Insights', icon: Lightbulb },
];

interface AppShellProps {
  page: PageId;
  onNavigate: (page: PageId) => void;
  children: React.ReactNode;
}

export default function AppShell({ page, onNavigate, children }: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const nav = (
    <nav className="flex flex-col gap-1">
      {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
        const active = page === id;
        return (
          <button
            key={id}
            onClick={() => {
              onNavigate(id);
              setMobileOpen(false);
            }}
            className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
              active
                ? 'bg-brand-50 text-brand-700'
                : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
            }`}
          >
            <Icon className={`h-[18px] w-[18px] shrink-0 ${active ? 'text-brand-600' : 'text-slate-400'}`} />
            {label}
          </button>
        );
      })}
    </nav>
  );

  return (
    <div className="min-h-screen bg-slate-100">
      {/* Sidebar - desktop */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-slate-200 bg-white px-4 py-6 lg:flex">
        <button onClick={() => onNavigate('dashboard')} className="mb-8 flex items-center gap-3 px-2 text-left">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-600 text-white">
            <ShoppingBag className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-bold leading-tight text-slate-900">Review Intelligence</p>
            <p className="text-xs text-slate-400">E-Commerce Analytics</p>
          </div>
        </button>
        {nav}
        <div className="mt-auto rounded-xl border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-center gap-2 text-amber-700">
            <FlaskConical className="h-4 w-4" />
            <p className="text-xs font-semibold">Foundation stage</p>
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-amber-700/80">
            ML backend (FastAPI + Scikit-learn) is not connected. All shown results are sample data.
          </p>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-40 flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 lg:hidden">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white">
            <ShoppingBag className="h-4 w-4" />
          </div>
          <p className="text-sm font-bold text-slate-900">Review Intelligence</p>
        </div>
        <button
          onClick={() => setMobileOpen((o) => !o)}
          className="rounded-lg p-2 text-slate-600 hover:bg-slate-100"
          aria-label="Toggle navigation"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-30 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/40" onClick={() => setMobileOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-slate-200 bg-white p-4 pt-6">
            {nav}
          </aside>
        </div>
      )}

      <main className="px-4 py-8 sm:px-6 lg:ml-64 lg:px-10">{children}</main>
    </div>
  );
}
