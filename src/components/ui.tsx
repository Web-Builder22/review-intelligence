import { FlaskConical } from 'lucide-react';
import type { ReactNode } from 'react';

interface PlaceholderBadgeProps {
  label?: string;
}

export function PlaceholderBadge({ label = 'Placeholder' }: PlaceholderBadgeProps) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-700">
      <FlaskConical className="h-3.5 w-3.5" />
      {label}
    </span>
  );
}

interface CardProps {
  title?: string;
  subtitle?: string;
  badge?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Card({ title, subtitle, badge, children, className = '' }: CardProps) {
  return (
    <div className={`rounded-2xl border border-slate-200 bg-white p-6 shadow-sm ${className}`}>
      {(title || badge) && (
        <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {badge}
        </div>
      )}
      {children}
    </div>
  );
}

export function EmptyState({ icon, title, message }: { icon: ReactNode; title: string; message: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-12 text-center">
      <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-slate-200 text-slate-400">
        {icon}
      </div>
      <p className="text-sm font-semibold text-slate-700">{title}</p>
      <p className="mt-1 max-w-sm text-sm text-slate-500">{message}</p>
    </div>
  );
}
