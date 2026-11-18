import { Router } from 'express';
import { prisma } from '../lib/prisma';
import { ApiError } from '../middleware/errorHandler';
import { validate } from '../middleware/validate';
import { createReviewSchema, listReviewsSchema } from '../schemas/review';

// IMPORTANT: no sentiment prediction happens here. When a review is created
// without explicit sentiment fields, they are stored as null until the
// FastAPI + Scikit-learn classification service is implemented (Phase 3+).
export const reviewsRouter = Router();

// GET /api/reviews
reviewsRouter.get('/', validate(listReviewsSchema, 'query'), async (req, res, next) => {
  try {
    const { limit, offset, sentiment } = (req as unknown as {
      query: { limit: number; offset: number; sentiment?: string };
    }).query;

    const [items, total] = await Promise.all([
      prisma.review.findMany({
        where: sentiment ? { sentiment } : undefined,
        orderBy: { createdAt: 'desc' },
        take: limit,
        skip: offset,
      }),
      prisma.review.count({ where: sentiment ? { sentiment } : undefined }),
    ]);

    res.json({ data: items, total, limit, offset });
  } catch (err) {
    next(err);
  }
});

// POST /api/reviews
reviewsRouter.post('/', validate(createReviewSchema), async (req, res, next) => {
  try {
    const { reviewText, sentiment, confidence, productName } = req.body as {
      reviewText: string;
      sentiment?: string;
      confidence?: number;
      productName?: string;
    };

    const review = await prisma.review.create({
      data: {
        reviewText,
        sentiment: sentiment ?? null,
        confidence: confidence ?? null,
        productName: productName ?? null,
      },
    });

    res.status(201).json({ data: review });
  } catch (err) {
    next(err);
  }
});

// GET /api/reviews/:id
reviewsRouter.get('/:id', async (req, res, next) => {
  try {
    const review = await prisma.review.findUnique({ where: { id: req.params.id } });
    if (!review) throw new ApiError(404, 'Review not found');
    res.json({ data: review });
  } catch (err) {
    next(err);
  }
});

// DELETE /api/reviews/:id
reviewsRouter.delete('/:id', async (req, res, next) => {
  try {
    const existing = await prisma.review.findUnique({ where: { id: req.params.id } });
    if (!existing) throw new ApiError(404, 'Review not found');
    await prisma.review.delete({ where: { id: req.params.id } });
    res.status(204).send();
  } catch (err) {
    next(err);
  }
});
