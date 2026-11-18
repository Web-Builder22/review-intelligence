import { GitCompare, FlaskConical } from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, PlaceholderBadge } from '@/components/ui';
import { MODEL_METRICS } from '@/data/placeholders';
import type { PageId } from '@/types/navigation';

interface ModelComparisonProps {
  onNavigate: (page: PageId) => void;
}

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs font-medium text-slate-500">
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${color} transition-all duration-500`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

const colors = ['bg-brand-500', 'bg-sky-500', 'bg-teal-500', 'bg-cyan-600'];

export default function ModelComparison({ onNavigate }: ModelComparisonProps) {
  return (
    <AppShell page="models" onNavigate={onNavigate}>
      <PageHeader
        title="Model Comparison"
        subtitle="Side-by-side performance of the four candidate sentiment classification algorithms."
        icon={<GitCompare className="h-6 w-6" />}
        actions={<PlaceholderBadge label="Sample data — not real training" />}
      />

      <div className="mb-6 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
        <FlaskConical className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
        <p className="text-sm leading-relaxed text-amber-800">
          All metrics shown below are <strong>sample placeholder values</strong>. No model training
          or evaluation has been performed yet — real numbers will appear here once the
          FastAPI + Scikit-learn backend is built.
        </p>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        {MODEL_METRICS.map((model, i) => (
          <Card key={model.name} className="animate-fade-in-up">
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl text-sm font-bold text-white ${colors[i % colors.length]}`}>
                  {i + 1}
                </div>
                <div>
                  <p className="text-sm font-bold text-slate-900">{model.name}</p>
                  <p className="text-xs text-slate-400">Supervised classification</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-slate-900">{(model.accuracy * 100).toFixed(1)}%</p>
                <p className="text-xs text-slate-400">accuracy</p>
              </div>
            </div>
            <div className="space-y-3.5">
              <MetricBar label="Accuracy" value={model.accuracy} color={colors[i % colors.length]} />
              <MetricBar label="Precision" value={model.precision} color={colors[i % colors.length]} />
              <MetricBar label="Recall" value={model.recall} color={colors[i % colors.length]} />
              <MetricBar label="F1 Score" value={model.f1} color={colors[i % colors.length]} />
            </div>
          </Card>
        ))}
      </div>

      <Card className="mt-6" title="Metrics Table" subtitle="Raw sample values for all four algorithms.">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[540px] text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs font-semibold uppercase tracking-wide text-slate-400">
                <th className="pb-3 pr-4">Algorithm</th>
                <th className="pb-3 pr-4">Accuracy</th>
                <th className="pb-3 pr-4">Precision</th>
                <th className="pb-3 pr-4">Recall</th>
                <th className="pb-3">F1 Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {MODEL_METRICS.map((m) => (
                <tr key={m.name} className="text-slate-700">
                  <td className="py-3 pr-4 font-semibold text-slate-900">{m.name}</td>
                  <td className="py-3 pr-4">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-3 pr-4">{(m.precision * 100).toFixed(1)}%</td>
                  <td className="py-3 pr-4">{(m.recall * 100).toFixed(1)}%</td>
                  <td className="py-3">{m.f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </AppShell>
  );
}
