import { useCallback, useRef, useState } from 'react';
import { Upload, FileSpreadsheet, CheckCircle2, AlertCircle, Lock, Loader2 } from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, PlaceholderBadge, EmptyState } from '@/components/ui';
import { getBulkFileInfo } from '@/services/api';
import type { BulkFileInfo } from '@/types';
import type { PageId } from '@/types/navigation';

interface BulkAnalysisProps {
  onNavigate: (page: PageId) => void;
}

function parseCsvPreview(file: File): Promise<{ rowCount: number; hasReviewColumn: boolean; hasRatingColumn: boolean }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Could not read the file.'));
    reader.onload = () => {
      const content = String(reader.result ?? '');
      const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
      const header = (lines[0] ?? '').toLowerCase();
      resolve({
        rowCount: Math.max(0, lines.length - 1),
        hasReviewColumn: /review|comment|text|feedback/.test(header),
        hasRatingColumn: /rating|score|stars/.test(header),
      });
    };
    reader.readAsText(file.slice(0, 1024 * 1024));
  });
}

export default function BulkAnalysis({ onNavigate }: BulkAnalysisProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fileInfo, setFileInfo] = useState<BulkFileInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setFileInfo(null);
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a .csv file containing customer reviews.');
      return;
    }
    setLoading(true);
    try {
      const { rowCount, hasReviewColumn, hasRatingColumn } = await parseCsvPreview(file);
      setFileInfo(getBulkFileInfo(file, rowCount, hasReviewColumn, hasRatingColumn));
    } catch {
      setError('Could not read the file. Make sure it is a valid CSV.');
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <AppShell page="bulk" onNavigate={onNavigate}>
      <PageHeader
        title="Bulk Review Analysis"
        subtitle="Upload a CSV of customer reviews for batch sentiment analysis. Actual prediction will run once the ML backend is connected."
        icon={<Upload className="h-6 w-6" />}
        actions={<PlaceholderBadge label="Upload ready — analysis pending" />}
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Upload CSV File" subtitle="The file should include a column with review text.">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files?.[0];
              if (file) void handleFile(file);
            }}
            onClick={() => inputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors ${
              dragOver
                ? 'border-brand-400 bg-brand-50'
                : 'border-slate-200 bg-slate-50 hover:border-brand-300 hover:bg-brand-50/50'
            }`}
          >
            {loading ? (
              <Loader2 className="mb-3 h-8 w-8 animate-spin text-brand-500" />
            ) : (
              <Upload className={`mb-3 h-8 w-8 ${dragOver ? 'text-brand-500' : 'text-slate-300'}`} />
            )}
            <p className="text-sm font-semibold text-slate-700">
              {loading ? 'Reading file…' : 'Drag & drop your CSV here'}
            </p>
            <p className="mt-1 text-sm text-slate-500">or click to browse files</p>
            <p className="mt-3 text-xs text-slate-400">Accepted format: .csv</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFile(file);
              e.target.value = '';
            }}
          />

          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </Card>

        <Card title="File Details" subtitle="Basic information about the uploaded file." badge={<PlaceholderBadge label="File info only" />}>
          {!fileInfo ? (
            <EmptyState
              icon={<FileSpreadsheet className="h-6 w-6" />}
              title="No file uploaded yet"
              message="Upload a CSV to see its filename, size, and detected columns."
            />
          ) : (
            <div className="animate-fade-in-up space-y-4">
              <div className="flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                <FileSpreadsheet className="h-8 w-8 shrink-0 text-emerald-600" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-bold text-slate-900">{fileInfo.name}</p>
                  <p className="text-xs text-slate-500">
                    {(fileInfo.sizeBytes / 1024).toFixed(1)} KB · {fileInfo.rowCount.toLocaleString()} data rows detected
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="flex items-center gap-2 rounded-xl border border-slate-100 bg-slate-50 p-3">
                  {fileInfo.hasReviewColumn ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
                  ) : (
                    <AlertCircle className="h-4 w-4 shrink-0 text-amber-500" />
                  )}
                  <p className="text-xs font-medium text-slate-600">
                    Review text column: {fileInfo.hasReviewColumn ? 'detected' : 'not detected'}
                  </p>
                </div>
                <div className="flex items-center gap-2 rounded-xl border border-slate-100 bg-slate-50 p-3">
                  {fileInfo.hasRatingColumn ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
                  ) : (
                    <AlertCircle className="h-4 w-4 shrink-0 text-amber-500" />
                  )}
                  <p className="text-xs font-medium text-slate-600">
                    Rating column: {fileInfo.hasRatingColumn ? 'detected' : 'not detected'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>

      <Card
        className="mt-6"
        title="Bulk Analysis Results"
        subtitle="Per-review sentiment and aggregated statistics will appear here after processing."
        badge={<PlaceholderBadge label="Awaiting ML backend" />}
      >
        <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-14 text-center">
          <Lock className="mb-3 h-8 w-8 text-slate-300" />
          <p className="text-sm font-semibold text-slate-700">Bulk processing not yet available</p>
          <p className="mt-1 max-w-md text-sm text-slate-500">
            This section will display sentiment counts, per-review classifications, and downloadable
            results once the FastAPI + Scikit-learn prediction service is implemented.
          </p>
        </div>
      </Card>
    </AppShell>
  );
}
