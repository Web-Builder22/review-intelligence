export type PageId = 'dashboard' | 'single' | 'bulk' | 'models' | 'insights';

export interface NavItem {
  id: PageId;
  label: string;
  icon: string;
  description: string;
}
