import { Router } from 'express';
import { prisma } from '../lib/prisma';
import { ApiError } from '../middleware/errorHandler';

// GET /api/models/performance
// Returns only metrics that were actually measured and stored (e.g. after a
// real training/evaluation run). If none exist yet, the response says so
// explicitly instead of serving hardcoded sample values as real results.
export const modelsRouter = Router();

const LATEST_PER_MODEL_WINDOW = 4; // max models tracked

modelsRouter.get('/performance', async (_req, res, next) => {
  try {
    const modelNames = await prisma.modelPerformance.findMany({
      distinct: ['modelName'],
      orderBy: { createdAt: 'desc' },
      select: { modelName: true },
    });

    if (modelNames.length === 0) {
      res.json({
        data: [],
        meta: {
          mlBackendConnected: false,
          note: 'No evaluated models yet. Metrics will appear here after a real training and evaluation run.',
        },
      });
      return;
    }

    // Latest evaluation row per model
    const rows = await Promise.all(
      modelNames.slice(0, LATEST_PER_MODEL_WINDOW).map(async ({ modelName }) => {
        const latest = await prisma.modelPerformance.findFirst({
          where: { modelName },
          orderBy: { createdAt: 'desc' },
        });
        return latest;
      })
    );

    const data = rows
      .filter((r): r is NonNullable<typeof r> => r !== null)
      .map((r) => ({
        id: r.id,
        modelName: r.modelName,
        accuracy: r.accuracy,
        precision: r.precision,
        recall: r.recall,
        f1Score: r.f1Score,
        createdAt: r.createdAt,
      }));

    res.json({ data, meta: { mlBackendConnected: true } });
  } catch (err) {
    next(err);
  }
});

// POST /api/models/performance
// Reserved for the future ML pipeline to record real evaluation results.
modelsRouter.post('/performance', async (req, res, next) => {
  try {
    const { modelName, accuracy, precision, recall, f1Score } = req.body as {
      modelName?: string;
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1Score?: number;
    };

    if (!modelName || typeof accuracy !== 'number' || typeof precision !== 'number' || typeof recall !== 'number' || typeof f1Score !== 'number') {
      throw new ApiError(400, 'modelName, accuracy, precision, recall and f1Score are required');
    }

    if ([accuracy, precision, recall, f1Score].some((v) => v < 0 || v > 1)) {
      throw new ApiError(400, 'Metrics must be between 0 and 1');
    }

    const record = await prisma.modelPerformance.create({
      data: { modelName, accuracy, precision, recall, f1Score },
    });

    res.status(201).json({ data: record });
  } catch (err) {
    next(err);
  }
});
