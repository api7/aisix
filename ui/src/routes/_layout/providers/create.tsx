import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';
import { ProviderForm } from '@/components/providers/provider-form';
import { Button } from '@/components/ui/button';
import type { Provider } from '@/lib/api/types';
import { useCreateProvider } from '@/lib/queries/providers';

export const Route = createFileRoute('/_layout/providers/create')({
  component: ProviderCreatePage,
});

function ProviderCreatePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createProvider = useCreateProvider();

  async function handleSubmit(data: Provider) {
    await createProvider.mutateAsync(data);
    navigate({ to: '/providers' });
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">{t('providers.title')}</h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate({ to: '/providers' })}
          aria-label={t('common.close')}
        >
          <X className="h-5 w-5" />
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl space-y-6">
          <div>
            <h2 className="text-base font-semibold">
              {t('providers.createTitle')}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {t('providers.createDesc')}
            </p>
          </div>

          <ProviderForm
            onSubmit={handleSubmit}
            isPending={createProvider.isPending}
            error={createProvider.error?.message}
            onCancel={() => navigate({ to: '/providers' })}
            submitLabel={t('providers.createProvider')}
          />
        </div>
      </div>
    </div>
  );
}
