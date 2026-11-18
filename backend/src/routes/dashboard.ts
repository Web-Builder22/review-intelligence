import { Router } from 'express';
import { prisma } from '../lib/prisma';

// GET /api/dashboard/stats
// Counts are computed live from stored reviews. Reviews that have not been
// classified yet (sentiment is null) are reported as "unclassified" and are
// NOT counted as neutral — no fake predictions are made here.
export const dashboardRouter = Router();

dashboardRouter.get('/stats', async (_req, res, next) => {
  try {
    const grouped = await prisma.review.groupBy({
      by: ['sentiment'],
      _count: { _all: true },
    });

    let positive = 0;
    let negative = 0;
    let neutral = 0;
    let unclassified = 0;

    for (const row of grouped) {
      switch (row.sentiment) {
        case 'positive':
          positive = row._count._all;
          break;
        case 'negative':
          negative = row._count._all;
          break;
        case 'neutral':
          neutral = row._count._all;
          break;
        default:
          unclassified += row._count._all;
      }
    }

    const totalReviews = positive + negative + neutral + unclassified;

    const recentReviews = await prisma.review.findMany({
      orderBy: { createdAt: 'desc' },
      take: 5,
      select: {
        id: true,
        reviewText: true,
        productName: true,
        sentiment: true,
        confidence: true,
        createdAt: true,
      },
    });

    res.json({
      data: {
        totalReviews,
        positive,
        negative,
        neutral,
        unclassified,
        recentReviews,
      },
      meta: {
        mlBackendConnected: false,
        note: 'Sentiment counts reflect stored classifications only. Reviews awaiting the future ML service are listed as unclassified.',
      },
    });
  } catch (err) {
    next(err);
  }
});
