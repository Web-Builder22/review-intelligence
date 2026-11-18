import { CheckCircle2, XCircle, MinusCircle, BarChart3, Cpu, ChevronRight, TrendingUp } from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, PlaceholderBadge } from '@/components/ui';
import { SentimentBadge, sentimentColor } from '@/components/SentimentBadge';
import { DASHBOARD_STATS, MODEL_METRICS } from '@/data/placeholders';
import type { PageId } from '@/types/navigation';

interface DashboardProps {
  onNavigate: (page: PageId) => void;
}

export default function Dashboard({ onNavigate }: DashboardProps) {
  const { totalReviews, positive, negative, neutral, recentActivity } = DASHBOARD_STATS;

  const distribution = [
    { label: 'Positive', value: positive, color: 'bg-emerald-500', icon: CheckCircle2, text: 'text-emerald-600' },
    { label: 'Negative', value: negative, color: 'bg-rose-500', icon: XCircle, text: 'text-rose-600' },
    { label: 'Neutral', value: neutral, color: 'bg-slate-400', icon: MinusCircle, text: 'text-slate-500' },
  ];

  const bestModel = MODEL_METRICS[0];

  return (
    <AppShell page="dashboard" onNavigate={onNavigate}>
      <PageHeader
        title="Dashboard"
        subtitle="Overview of customer review analysis across your e-commerce catalog."
        icon={<BarChart3 className="h-6 w-6" />}
        actions={<PlaceholderBadge label="Sample data" />}
      />

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <div className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <p className="text-sm font-medium text-slate-500">Total Reviews Analyzed</p>
          <p className="mt-2 text-3xl font-bold tracking-tight text-slate-900">{totalReviews.toLocaleString()}</p>
          <div className="mt-3 flex items-center gap-1.5 text-xs font-medium text-emerald-600">
            <TrendingUp className="h-3.5 w-3.5" />
            +8.2% vs. last month
          </div>
        </div>
        {distribution.map(({ label, value, icon: Icon, text }) => (
          <div key={label} className="animate-fade-in-up rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-slate-500">{label} Reviews</p>
              <Icon className={`h-4 w-4 ${text}`} />
            </div>
            <p className="mt-2 text-3xl font-bold tracking-tight text-slate-900">{value.toLocaleString()}</p>
            <div className="mt-3 text-xs font-medium text-slate-500">
              {((value / totalReviews) * 100).toFixed(1)}% of total
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Sentiment Distribution" subtitle="Overall breakdown across all analyzed reviews.">
          <div className="flex h-3 w-full overflow-hidden rounded-full">
            {distribution.map(({ label, value, color }) => (
              <div
                key={label}
                className={color}
                style={{ width: `${(value / totalReviews) * 100}%` }}
                title={`${label}: ${value.toLocaleString()}`}
              />
            ))}
          </div>
          <div className="mt-5 grid grid-cols-3 gap-4">
            {distribution.map(({ label, value, color, text }) => (
              <div key={label} className="rounded-xl border border-slate-100 bg-slate-50 p-4">
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${color}`} />
                  <p className="text-xs font-semibold text-slate-600">{label}</p>
                </div>
                <p className={`mt-1.5 text-xl font-bold ${text}`}>{((value / totalReviews) * 100).toFixed(1)}%</p>
                <p className="text-xs text-slate-400">{value.toLocaleString()} reviews</p>
              </div>
            ))}
          </div>
        </Card>

        <Card
          title="Recent Review Activity"
          subtitle="Latest customer reviews detected in the feed."
        >
          <ul className="divide-y divide-slate-100">
            {recentActivity.map((item) => (
              <li key={item.id} className="flex items-start justify-between gap-4 py-3 first:pt-0 last:pb-0">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-800">{item.product}</p>
                  <p className="mt-0.5 truncate text-sm text-slate-500">{item.snippet}</p>
                  <p className="mt-1 text-xs text-slate-400">{item.date}</p>
                </div>
                <SentimentBadge sentiment={item.sentiment} />
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card
        className="mt-6"
        title="Model Performance Overview"
        subtitle="Summary of the best-performing algorithm. Full comparison is on the Model Comparison page."
        badge={<PlaceholderBadge label="Placeholders" />}
      >
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { label: 'Best Model', value: bestModel.name },
            { label: 'Accuracy', value: `${(bestModel.accuracy * 100).toFixed(1)}%` },
            { label: 'F1 Score', value: bestModel.f1.toFixed(2) },
            { label: 'Algorithms Tracked', value: String(MODEL_METRICS.length) },
          ].map(({ label, value }) => (
            <div key={label} className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</p>
              <p className="mt-1.5 text-lg font-bold text-slate-900">{value}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-2">
          {MODEL_METRICS.map((m) => (
            <div key={m.name} className="flex-1">
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full ${sentimentColor.neutral === 'bg-slate-400' ? 'bg-brand-500' : ''}`}
                  style={{ width: `${m.accuracy * 100}%` }}
                />
              </div>
              <p className="mt-1 truncate text-[11px] text-slate-400">{m.name}</p>
            </div>
          ))}
        </div>
        <button
          onClick={() => onNavigate('models')}
          className="mt-5 inline-flex items-center gap-1 text-sm font-semibold text-brand-600 hover:text-brand-700"
        >
          View full model comparison
          <ChevronRight className="h-4 w-4" />
        </button>
      </Card>

      <div className="mt-6 flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-400">
        <Cpu className="h-3.5 w-3.5 shrink-0" />
        All figures on this page are sample placeholders. The FastAPI + Scikit-learn backend has not been implemented yet.
      </div>
    </AppShell>
  );
}
