import { Link, useRouterState } from '@tanstack/react-router';
import {
  Boxes,
  ChevronsUpDown,
  KeyRound,
  LayoutDashboard,
  Settings,
  Zap,
} from 'lucide-react';

import { useAdminKey } from '@/hooks/use-admin-key';
import { cn } from '@/lib/utils';

const NAV_GROUPS = [
  {
    label: 'PLATFORM',
    items: [
      { to: '/playground', label: 'Playground', icon: LayoutDashboard },
      { to: '/models', label: 'Models', icon: Boxes },
      { to: '/apikeys', label: 'API Keys', icon: KeyRound },
    ],
  },
  {
    label: 'GENERAL',
    items: [{ to: '/settings', label: 'Settings', icon: Settings }],
  },
] as const;

function NavItem({
  to,
  label,
  icon: Icon,
}: {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isActive = pathname === to || pathname.startsWith(to + '/');

  return (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors',
        isActive
          ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
          : 'text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground',
      )}
    >
      <Icon className="h-4 w-4 flex-none" />
      {label}
    </Link>
  );
}

export function DashboardSidebar() {
  const { key, openModal } = useAdminKey();
  const maskedKey = key ? `API Key has been set` : 'No API Key has been set';

  return (
    <div className="flex h-full flex-col bg-sidebar">
      {/* Header */}
      <div className="flex h-14 items-center gap-2.5 border-b border-sidebar-border px-4">
        <div className="flex h-8 w-8 flex-none items-center justify-center rounded-lg bg-primary">
          <Zap className="h-4 w-4 text-primary-foreground" strokeWidth={2.5} />
        </div>
        <span className="text-[15px] font-semibold tracking-tight text-sidebar-foreground">
          AISIX
        </span>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto p-2">
        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="mb-3">
            <p className="mb-1 px-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              {group.label}
            </p>
            <div className="space-y-0.5">
              {group.items.map((item) => (
                <NavItem key={item.to} {...item} />
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer — click to open admin key modal */}
      <button
        type="button"
        onClick={openModal}
        className="flex h-14 w-full cursor-pointer items-center gap-2.5 border-t border-sidebar-border px-4 transition-colors hover:bg-sidebar-accent/60"
      >
        <div className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-primary text-[13px] font-semibold text-primary-foreground">
          A
        </div>
        <div className="min-w-0 flex-1 text-left">
          <p className="truncate text-[13px] leading-tight font-medium text-sidebar-foreground">
            Admin User
          </p>
          <p className="truncate text-[11px] text-muted-foreground">
            {maskedKey}
          </p>
        </div>
        <ChevronsUpDown className="h-4 w-4 flex-none text-muted-foreground" />
      </button>
    </div>
  );
}
