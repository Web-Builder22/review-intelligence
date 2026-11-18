import { Request, Response, NextFunction } from 'express';

export class ApiError extends Error {
  public readonly status: number;
  public readonly details?: unknown;

  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

export const notFoundHandler = (req: Request, _res: Response, next: NextFunction): void => {
  next(new ApiError(404, `Route not found: ${req.method} ${req.originalUrl}`));
};

export function isProduction(): boolean {
  return process.env.NODE_ENV === 'production';
}

// Centralized error handler: converts any thrown error into a JSON response.
// Unexpected errors are logged but never leak internals to the client in production.
export const errorHandler = (
  err: unknown,
  _req: Request,
  res: Response,
  _next: NextFunction
): void => {
  if (err instanceof ApiError) {
    res.status(err.status).json({
      error: err.message,
      ...(err.details !== undefined ? { details: err.details } : {}),
    });
    return;
  }

  const isProd = isProduction();
  if (!isProd) {
    console.error('[Unhandled error]', err);
  }

  res.status(500).json({
    error: isProd ? 'Internal server error' : 'Internal server error',
  });
};
