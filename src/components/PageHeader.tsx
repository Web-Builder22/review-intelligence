import type { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  subtitle: string;
  icon?: ReactNode;
  actions?: ReactNode;
}

export default function PageHeader({ title, subtitle, icon, actions }: PageHeaderProps) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        {icon && (
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white shadow-sm">
            {icon}
          </div>
        )}
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">{title}</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-500">{subtitle}</p>
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-3">{actions}</div>}
    </div>
  );
}
