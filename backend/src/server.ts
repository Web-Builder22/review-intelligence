import 'dotenv/config';
import { createApp } from './app';
import { prisma } from './lib/prisma';

const port = Number(process.env.PORT ?? 4000);

const app = createApp();

const server = app.listen(port, () => {
  console.log(`Review Intelligence API listening on http://localhost:${port}`);
});

const shutdown = async (signal: string) => {
  console.log(`\n${signal} received, shutting down…`);
  server.close(async () => {
    await prisma.$disconnect();
    process.exit(0);
  });
};

process.on('SIGINT', () => void shutdown('SIGINT'));
process.on('SIGTERM', () => void shutdown('SIGTERM'));
