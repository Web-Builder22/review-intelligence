import { Request, Response, NextFunction } from 'express';
import { ZodSchema, ZodError } from 'zod';
import { ApiError } from './errorHandler';

type Target = 'body' | 'query' | 'params';

export const validate = (schema: ZodSchema, target: Target = 'body') =>
  (req: Request, _res: Response, next: NextFunction): void => {
    try {
      req[target] = schema.parse(req[target]);
      next();
    } catch (err) {
      if (err instanceof ZodError) {
        next(new ApiError(400, 'Validation failed', err.flatten().fieldErrors));
        return;
      }
      next(err);
    }
  };
