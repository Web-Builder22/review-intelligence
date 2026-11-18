import { useState } from 'react';
import type { PageId } from '@/types/navigation';
import Dashboard from '@/pages/Dashboard';
import SingleAnalysis from '@/pages/SingleAnalysis';
import BulkAnalysis from '@/pages/BulkAnalysis';
import ModelComparison from '@/pages/ModelComparison';
import CustomerInsights from '@/pages/CustomerInsights';

function readHash(): PageId {
  const hash = window.location.hash.replace('#/', '').replace('#', '');
  const valid: PageId[] = ['dashboard', 'single', 'bulk', 'models', 'insights'];
  return (valid as string[]).includes(hash) ? (hash as PageId) : 'dashboard';
}

export default function App() {
  const [page, setPage] = useState<PageId>(readHash);

  const navigate = (next: PageId) => {
    window.location.hash = `/${next}`;
    setPage(next);
    window.scrollTo({ top: 0 });
  };

  switch (page) {
    case 'single':
      return <SingleAnalysis onNavigate={navigate} />;
    case 'bulk':
      return <BulkAnalysis onNavigate={navigate} />;
    case 'models':
      return <ModelComparison onNavigate={navigate} />;
    case 'insights':
      return <CustomerInsights onNavigate={navigate} />;
    default:
      return <Dashboard onNavigate={navigate} />;
  }
}
