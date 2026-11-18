import type { Sentiment } from '@/types';

const styles: Record<Sentiment, { bg: string; text: string; dot: string; label: string }> = {
  positive: { bg: 'bg-emerald-50 border-emerald-200', text: 'text-emerald-700', dot: 'bg-emerald-500', label: 'Positive' },
  negative: { bg: 'bg-rose-50 border-rose-200', text: 'text-rose-700', dot: 'bg-rose-500', label: 'Negative' },
  neutral: { bg: 'bg-slate-50 border-slate-200', text: 'text-slate-600', dot: 'bg-slate-400', label: 'Neutral' },
};

export function SentimentBadge({ sentiment }: { sentiment: Sentiment }) {
  const s = styles[sentiment];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${s.bg} ${s.text}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}

export const sentimentColor: Record<Sentiment, string> = {
  positive: 'bg-emerald-500',
  negative: 'bg-rose-500',
  neutral: 'bg-slate-400',
};
