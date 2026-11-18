import { z } from 'zod';

export const createReviewSchema = z.object({
  reviewText: z.string().trim().min(1, 'reviewText is required').max(5000, 'reviewText must be at most 5000 characters'),
  sentiment: z.enum(['positive', 'negative', 'neutral']).optional(),
  confidence: z.number().min(0).max(1).optional(),
  productName: z.string().trim().max(200).optional(),
});

export const listReviewsSchema = z.object({
  limit: z.coerce.number().int().min(1).max(100).default(20),
  offset: z.coerce.number().int().min(0).default(0),
  sentiment: z.enum(['positive', 'negative', 'neutral']).optional(),
});

export type CreateReviewInput = z.infer<typeof createReviewSchema>;
export type ListReviewsInput = z.infer<typeof listReviewsSchema>;
