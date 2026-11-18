import type { AnalysisResult, BulkFileInfo } from '@/types';

/**
 * Frontend service layer for the future Python FastAPI + Scikit-learn
 * backend. Every function below is a stub that returns clearly labeled
 * placeholder values — no machine learning prediction happens here yet.
 *
 * When the FastAPI service is ready, replace the bodies of these
 * functions with real HTTP calls (e.g. using fetch against
 * `API_BASE_URL`) and remove the placeholder responses.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function analyzeSingleReview(_text: string): Promise<AnalysisResult> {
  // TODO: replace with `POST ${API_BASE_URL}/analyze` once the ML service exists.
  await wait(600);
  return {
    sentiment: 'neutral',
    confidence: 0,
    explanation:
      'Placeholder result. No machine learning backend is connected yet — the FastAPI + Scikit-learn service has not been implemented.',
  };
}

export async function analyzeBulkReviews(_file: File): Promise<never> {
  // TODO: replace with `POST ${API_BASE_URL}/analyze/bulk` once the ML service exists.
  throw new Error(
    'Bulk analysis is not available yet. The FastAPI + Scikit-learn backend has not been implemented.'
  );
}

export function getBulkFileInfo(file: File, rowCount: number, hasReviewColumn: boolean, hasRatingColumn: boolean): BulkFileInfo {
  return {
    name: file.name,
    sizeBytes: file.size,
    rowCount,
    hasReviewColumn,
    hasRatingColumn,
  };
}

export { API_BASE_URL };
