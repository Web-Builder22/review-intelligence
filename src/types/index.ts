export type Sentiment = 'positive' | 'negative' | 'neutral';

export interface DashboardStats {
  totalReviews: number;
  positive: number;
  negative: number;
  neutral: number;
  recentActivity: { id: number; product: string; snippet: string; sentiment: Sentiment; date: string }[];
}

export interface AnalysisResult {
  sentiment: Sentiment;
  confidence: number;
  explanation: string;
}

export interface BulkFileInfo {
  name: string;
  sizeBytes: number;
  rowCount: number;
  hasReviewColumn: boolean;
  hasRatingColumn: boolean;
}

export interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface InsightItem {
  topic: string;
  count: number;
  share: number;
}
