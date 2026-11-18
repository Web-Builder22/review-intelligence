import type { DashboardStats, InsightItem, ModelMetrics } from '@/types';

export const DASHBOARD_STATS: DashboardStats = {
  totalReviews: 12480,
  positive: 7240,
  negative: 3110,
  neutral: 2130,
  recentActivity: [
    { id: 1, product: 'Wireless Earbuds Pro', snippet: 'Battery life is amazing, lasts two days straight.', sentiment: 'positive', date: '2026-11-17 09:12' },
    { id: 2, product: 'Smart Watch X2', snippet: 'Stopped syncing after a week. Very frustrated.', sentiment: 'negative', date: '2026-11-17 08:45' },
    { id: 3, product: 'USB-C Charging Cable', snippet: 'It is a cable. It charges. Does the job.', sentiment: 'neutral', date: '2026-11-16 21:30' },
    { id: 4, product: 'Desk Lamp LED', snippet: 'Beautiful light, easy assembly, great value.', sentiment: 'positive', date: '2026-11-16 18:02' },
    { id: 5, product: 'Bluetooth Speaker Mini', snippet: 'Sound distorts at high volume, otherwise fine.', sentiment: 'negative', date: '2026-11-16 15:47' },
  ],
};

export const MODEL_METRICS: ModelMetrics[] = [
  { name: 'Linear SVM', accuracy: 0.91, precision: 0.9, recall: 0.89, f1: 0.9 },
  { name: 'Logistic Regression', accuracy: 0.89, precision: 0.88, recall: 0.87, f1: 0.87 },
  { name: 'Random Forest', accuracy: 0.86, precision: 0.85, recall: 0.84, f1: 0.84 },
  { name: 'Multinomial Naive Bayes', accuracy: 0.84, precision: 0.83, recall: 0.82, f1: 0.82 },
];

export const POSITIVE_TOPICS: InsightItem[] = [
  { topic: 'Fast delivery', count: 1840, share: 0.25 },
  { topic: 'Good value for money', count: 1520, share: 0.21 },
  { topic: 'Battery life', count: 1190, share: 0.16 },
  { topic: 'Easy setup', count: 940, share: 0.13 },
  { topic: 'Build quality', count: 760, share: 0.1 },
];

export const COMMON_COMPLAINTS: InsightItem[] = [
  { topic: 'Stopped working within a month', count: 980, share: 0.31 },
  { topic: 'Long shipping time', count: 720, share: 0.23 },
  { topic: 'Inaccurate product description', count: 610, share: 0.19 },
  { topic: 'Poor packaging', count: 480, share: 0.15 },
  { topic: 'Unresponsive seller support', count: 390, share: 0.12 },
];

export const FEEDBACK_TRENDS: { month: string; positive: number; negative: number }[] = [
  { month: 'Jun', positive: 380, negative: 210 },
  { month: 'Jul', positive: 420, negative: 190 },
  { month: 'Aug', positive: 455, negative: 205 },
  { month: 'Sep', positive: 510, negative: 230 },
  { month: 'Oct', positive: 560, negative: 205 },
  { month: 'Nov', positive: 615, negative: 175 },
];
