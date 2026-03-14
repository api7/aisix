import { Link, useRouterState } from '@tanstack/react-router';
import {
  Boxes,
  ChevronsUpDown,
  Languages,
  KeyRound,
  LayoutDashboard,
  Monitor,
  Moon,
  Settings,
  Sun,
  Zap,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useTheme } from '@/components/theme-provider';
import { useAdminKey } from '@/hooks/use-admin-key';
import { cn } from '@/lib/utils';

const NAV_GROUPS = [
  {
    labelKey: 'nav.platform',
    items: [
      { to: '/playground', labelKey: 'nav.playground', icon: LayoutDashboard },
      { to: '/models', labelKey: 'nav.models', icon: Boxes },
      { to: '/apikeys', labelKey: 'nav.apiKeys', icon: KeyRound },
    ],
  },
  {
    labelKey: 'nav.general',
    items: [{ to: '/settings', labelKey: 'nav.settings', icon: Settings }],
  },
] as const;

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'zh-CN', label: '简体中文' },
] as const;
const LANG_STORAGE_KEY = 'aisix-ui-language';

function NavItem({
  to,
  labelKey,
  icon: Icon,
}: {
  to: string;
  labelKey: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  const { t } = useTranslation();
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
      {t(labelKey)}
    </Link>
  );
}

export function DashboardSidebar() {
  const { t, i18n } = useTranslation();
  const { theme, setTheme } = useTheme();
  const { key, openModal } = useAdminKey();
  const [themeOpen, setThemeOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const themeRef = useRef<HTMLDivElement>(null);
  const langRef = useRef<HTMLDivElement>(null);
  const maskedKey = key ? t('sidebar.apiKeySet') : t('sidebar.noApiKey');

  useEffect(() => {
    if (!langOpen && !themeOpen) return;

    function onDown(e: MouseEvent) {
      if (!themeRef.current?.contains(e.target as Node)) {
        setThemeOpen(false);
      }
      if (!langRef.current?.contains(e.target as Node)) {
        setLangOpen(false);
      }
    }

    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [langOpen, themeOpen]);

  const currentLanguage =
    LANGUAGES.find((item) => item.code === i18n.resolvedLanguage)?.code ?? 'en';

  return (
    <div className="flex h-full flex-col bg-sidebar">
      {/* Header */}
      <div className="flex h-14 items-center gap-2.5 border-b border-sidebar-border px-4">
        <div className="flex h-8 w-8 flex-none items-center justify-center rounded-lg bg-primary">
          <Zap className="h-4 w-4 text-primary-foreground" strokeWidth={2.5} />
        </div>
        <span className="text-[15px] font-semibold tracking-tight text-sidebar-foreground">
          {t('sidebar.appName')}
        </span>
        <div className="relative ml-auto" ref={themeRef}>
          <button
            type="button"
            onClick={() => setThemeOpen((v) => !v)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-sidebar-border text-sidebar-foreground transition-colors hover:bg-sidebar-accent/60"
            aria-label={t('sidebar.theme')}
            title={t('sidebar.theme')}
          >
            {theme === 'dark' ? (
              <Moon className="h-4 w-4" />
            ) : theme === 'light' ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Monitor className="h-4 w-4" />
            )}
          </button>

          {themeOpen && (
            <div className="absolute top-full right-0 z-50 mt-1 w-36 overflow-hidden rounded-lg border border-sidebar-border bg-sidebar shadow-[0_6px_24px_rgba(0,0,0,0.1)]">
              <button
                type="button"
                className={cn(
                  'flex h-8 w-full items-center gap-2 px-3 text-left text-xs text-sidebar-foreground hover:bg-sidebar-accent/60',
                  theme === 'system' &&
                    'bg-sidebar-accent font-medium text-sidebar-accent-foreground',
                )}
                onClick={() => {
                  setTheme('system');
                  setThemeOpen(false);
                }}
              >
                <Monitor className="h-3.5 w-3.5" />
                {t('sidebar.themeSystem')}
              </button>
              <button
                type="button"
                className={cn(
                  'flex h-8 w-full items-center gap-2 px-3 text-left text-xs text-sidebar-foreground hover:bg-sidebar-accent/60',
                  theme === 'light' &&
                    'bg-sidebar-accent font-medium text-sidebar-accent-foreground',
                )}
                onClick={() => {
                  setTheme('light');
                  setThemeOpen(false);
                }}
              >
                <Sun className="h-3.5 w-3.5" />
                {t('sidebar.themeLight')}
              </button>
              <button
                type="button"
                className={cn(
                  'flex h-8 w-full items-center gap-2 px-3 text-left text-xs text-sidebar-foreground hover:bg-sidebar-accent/60',
                  theme === 'dark' &&
                    'bg-sidebar-accent font-medium text-sidebar-accent-foreground',
                )}
                onClick={() => {
                  setTheme('dark');
                  setThemeOpen(false);
                }}
              >
                <Moon className="h-3.5 w-3.5" />
                {t('sidebar.themeDark')}
              </button>
            </div>
          )}
        </div>
        <div className="relative" ref={langRef}>
          <button
            type="button"
            onClick={() => setLangOpen((v) => !v)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-sidebar-border text-sidebar-foreground transition-colors hover:bg-sidebar-accent/60"
            aria-label={t('sidebar.language')}
            title={t('sidebar.language')}
          >
            <Languages className="h-4 w-4" />
          </button>

          {langOpen && (
            <div className="absolute top-full right-0 z-50 mt-1 w-36 overflow-hidden rounded-lg border border-sidebar-border bg-sidebar shadow-[0_6px_24px_rgba(0,0,0,0.1)]">
              {LANGUAGES.map((item) => (
                <button
                  key={item.code}
                  type="button"
                  className={cn(
                    'flex h-8 w-full items-center px-3 text-left text-xs text-sidebar-foreground hover:bg-sidebar-accent/60',
                    currentLanguage === item.code &&
                      'bg-sidebar-accent font-medium text-sidebar-accent-foreground',
                  )}
                  onClick={() => {
                    if (typeof window !== 'undefined') {
                      window.localStorage.setItem(LANG_STORAGE_KEY, item.code);
                    }
                    i18n.changeLanguage(item.code);
                    setLangOpen(false);
                  }}
                >
                  {item.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto p-2">
        {NAV_GROUPS.map((group) => (
          <div key={group.labelKey} className="mb-3">
            <p className="mb-1 px-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              {t(group.labelKey)}
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
            {t('sidebar.adminUser')}
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
