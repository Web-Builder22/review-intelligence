import { useState } from 'react';
import { Search, Sparkles, Info, Loader2 } from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, PlaceholderBadge } from '@/components/ui';
import { SentimentBadge } from '@/components/SentimentBadge';
import { analyzeSingleReview } from '@/services/api';
import type { AnalysisResult } from '@/types';
import type { PageId } from '@/types/navigation';

const EXAMPLES = [
  'This headphones sound quality is fantastic and the battery lasts forever.',
  'Arrived two weeks late and the box was crushed. Very disappointed.',
  'The package arrived on Tuesday.',
];

interface SingleAnalysisProps {
  onNavigate: (page: PageId) => void;
}

export default function SingleAnalysis({ onNavigate }: SingleAnalysisProps) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim() || loading) return;
    setLoading(true);
    setResult(null);
    const placeholder = await analyzeSingleReview(text);
    setResult(placeholder);
    setLoading(false);
  };

  const confidenceColor = (c: number) =>
    c >= 75 ? 'bg-emerald-500' : c >= 40 ? 'bg-amber-500' : 'bg-slate-400';

  return (
    <AppShell page="single" onNavigate={onNavigate}>
      <PageHeader
        title="Single Review Analysis"
        subtitle="Enter one customer review to classify its sentiment. Results are placeholders until the ML backend is connected."
        icon={<Search className="h-6 w-6" />}
        actions={<PlaceholderBadge label="Placeholder results" />}
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Review Text" subtitle="Paste or type the customer review below.">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. The product quality exceeded my expectations, shipping was fast too."
            rows={8}
            className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-relaxed text-slate-800 placeholder:text-slate-400 focus:border-brand-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-brand-100"
          />
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400">Try an example:</span>
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => setText(ex)}
                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-600 transition-colors hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
              >
                Example {i + 1}
              </button>
            ))}
          </div>
          <div className="mt-5 flex items-center justify-between gap-4">
            <p className="text-xs text-slate-400">{text.trim().length.toLocaleString()} characters</p>
            <button
              onClick={handleAnalyze}
              disabled={!text.trim() || loading}
              className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
          </div>
        </Card>

        <Card
          title="Analysis Result"
          subtitle="Sentiment classification with confidence and explanation."
          badge={<PlaceholderBadge />}
        >
          {loading && (
            <div className="flex animate-pulse flex-col gap-4 py-4">
              <div className="h-8 w-32 rounded-lg bg-slate-100" />
              <div className="h-3 w-full rounded-full bg-slate-100" />
              <div className="h-20 w-full rounded-xl bg-slate-100" />
            </div>
          )}

          {!loading && !result && (
            <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-12 text-center">
              <Sparkles className="mb-3 h-8 w-8 text-slate-300" />
              <p className="text-sm font-semibold text-slate-700">No analysis yet</p>
              <p className="mt-1 max-w-xs text-sm text-slate-500">
                Enter a review and press Analyze to see the sentiment classification.
              </p>
            </div>
          )}

          {!loading && result && (
            <div className="animate-fade-in-up space-y-5">
              <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 p-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Detected Sentiment</p>
                  <div className="mt-2">
                    <SentimentBadge sentiment={result.sentiment} />
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Confidence</p>
                  <p className="mt-1 text-3xl font-bold text-slate-900">
                    {result.confidence > 0 ? `${result.confidence.toFixed(1)}%` : '—'}
                  </p>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between text-xs font-medium text-slate-500">
                  <span>Confidence score</span>
                  <span>{result.confidence.toFixed(0)}%</span>
                </div>
                <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${confidenceColor(result.confidence)}`}
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
              </div>

              <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <p className="text-sm leading-relaxed text-amber-800">{result.explanation}</p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </AppShell>
  );
}
