import express, { Express } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import { healthRouter } from './routes/health';
import { reviewsRouter } from './routes/reviews';
import { dashboardRouter } from './routes/dashboard';
import { modelsRouter } from './routes/models';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';

const allowedOrigins = (process.env.CORS_ORIGIN ?? 'http://localhost:5173')
  .split(',')
  .map((o) => o.trim())
  .filter(Boolean);

const corsOptions: cors.CorsOptions = {
  origin(origin, callback) {
    // Allow requests with no Origin header (curl, server-to-server, same-origin).
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
      return;
    }
    callback(new Error(`Origin ${origin} not allowed by CORS`));
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
};

export function createApp(): Express {
  const app = express();

  app.use(helmet());
  app.use(cors(corsOptions));
  app.use(express.json({ limit: '1mb' }));

  app.use('/api/health', healthRouter);
  app.use('/api/reviews', reviewsRouter);
  app.use('/api/dashboard', dashboardRouter);
  app.use('/api/models', modelsRouter);

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}
