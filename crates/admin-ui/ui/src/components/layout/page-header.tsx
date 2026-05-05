import { Menu } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useSidebar } from '@/hooks/use-sidebar';

interface PageHeaderProps {
  children: React.ReactNode;
}

/**
 * Shared page header shell: fixed height, bottom border, horizontal padding.
 * Renders a mobile sidebar toggle automatically; page-specific content goes in children.
 */
export function PageHeader({ children }: PageHeaderProps) {
  const { toggle } = useSidebar();

  return (
    <header className="flex h-16 shrink-0 items-center gap-3 border-b px-6">
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden"
        onClick={toggle}
        aria-label="Toggle navigation"
      >
        <Menu className="h-5 w-5" />
      </Button>
      {children}
    </header>
  );
}
