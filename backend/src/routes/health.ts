import { Router } from 'express';
import { prisma } from '../lib/prisma';

// GET /api/health
// Confirms the API is up and the database connection works.
export const healthRouter = Router();

healthRouter.get('/', async (_req, res, next) => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.json({ status: 'ok', database: 'connected', timestamp: new Date().toISOString() });
  } catch (err) {
    next(err);
  }
});
