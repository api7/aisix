import { createFileRoute } from '@tanstack/react-router';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';

export const Route = createFileRoute('/_layout/settings')({
  component: SettingsPage,
});

function SettingsPage() {
  const { t } = useTranslation();
  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">{t('settings.title')}</h1>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl rounded-xl border bg-card p-6 text-sm text-muted-foreground">
          {t('settings.comingSoon')}
        </div>
      </div>
    </div>
  );
}
