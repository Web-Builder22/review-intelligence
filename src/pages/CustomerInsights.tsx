import { Lightbulb, ThumbsUp, ThumbsDown, TrendingUp } from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, PlaceholderBadge } from '@/components/ui';
import { POSITIVE_TOPICS, COMMON_COMPLAINTS, FEEDBACK_TRENDS } from '@/data/placeholders';
import type { PageId } from '@/types/navigation';

interface CustomerInsightsProps {
  onNavigate: (page: PageId) => void;
}

function TopicList({ items, color, icon }: { items: { topic: string; count: number; share: number }[]; color: string; icon: 'up' | 'down' }) {
  const Icon = icon === 'up' ? ThumbsUp : ThumbsDown;
  const max = items[0]?.count || 1;
  return (
    <ul className="space-y-4">
      {items.map((item) => (
        <li key={item.topic}>
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <Icon className={`h-4 w-4 shrink-0 ${icon === 'up' ? 'text-emerald-500' : 'text-rose-500'}`} />
              <p className="text-sm font-medium text-slate-700">{item.topic}</p>
            </div>
            <p className="shrink-0 text-xs font-semibold text-slate-500">
              {item.count.toLocaleString()} ({Math.round(item.share * 100)}%)
            </p>
          </div>
          <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full ${color} transition-all duration-500`}
              style={{ width: `${(item.count / max) * 100}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function CustomerInsights({ onNavigate }: CustomerInsightsProps) {
  const maxPositive = Math.max(...FEEDBACK_TRENDS.map((t) => t.positive));

  return (
    <AppShell page="insights" onNavigate={onNavigate}>
      <PageHeader
        title="Customer Insights"
        subtitle="Recurring themes extracted from review text — what customers praise, complain about, and how trends shift over time."
        icon={<Lightbulb className="h-6 w-6" />}
        actions={<PlaceholderBadge label="Sample data" />}
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <Card
          title="Common Positive Topics"
          subtitle="Themes most frequently praised in customer reviews."
          badge={<PlaceholderBadge />}
        >
          <TopicList items={POSITIVE_TOPICS} color="bg-emerald-500" icon="up" />
        </Card>

        <Card
          title="Common Complaints"
          subtitle="Issues customers raise most often in negative reviews."
          badge={<PlaceholderBadge />}
        >
          <TopicList items={COMMON_COMPLAINTS} color="bg-rose-500" icon="down" />
        </Card>
      </div>

      <Card
        className="mt-6"
        title="Customer Feedback Trends"
        subtitle="Monthly volume of positive vs. negative feedback over the last six months."
        badge={<PlaceholderBadge />}
      >
        <div className="flex h-56 items-end gap-4 sm:gap-6">
          {FEEDBACK_TRENDS.map((t) => (
            <div key={t.month} className="flex flex-1 flex-col items-center gap-2">
              <div className="flex h-full w-full items-end justify-center gap-1.5">
                <div
                  className="w-full max-w-[22px] rounded-t-md bg-emerald-500 transition-all duration-500"
                  style={{ height: `${(t.positive / maxPositive) * 100}%` }}
                  title={`Positive: ${t.positive}`}
                />
                <div
                  className="w-full max-w-[22px] rounded-t-md bg-rose-400 transition-all duration-500"
                  style={{ height: `${(t.negative / maxPositive) * 100}%` }}
                  title={`Negative: ${t.negative}`}
                />
              </div>
              <p className="text-xs font-medium text-slate-500">{t.month}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-center gap-6 text-xs font-medium text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Positive
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-rose-400" /> Negative
          </span>
        </div>
        <div className="mt-5 flex items-center gap-2 rounded-xl border border-slate-100 bg-slate-50 p-4">
          <TrendingUp className="h-4 w-4 shrink-0 text-emerald-600" />
          <p className="text-sm text-slate-600">
            Sample trend: positive feedback volume is rising while complaints are declining month over month.
          </p>
        </div>
      </Card>
    </AppShell>
  );
}
