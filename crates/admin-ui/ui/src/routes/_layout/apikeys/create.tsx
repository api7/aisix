import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { ApiKeyForm } from '@/components/apikeys/apikey-form';
import { PageHeader } from '@/components/layout/page-header';
import { Button } from '@/components/ui/button';
import type { ApiKey } from '@/lib/api/types';
import { useCreateApiKey } from '@/lib/queries/apikeys';

export const Route = createFileRoute('/_layout/apikeys/create')({
  component: ApiKeyCreatePage,
});

function ApiKeyCreatePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createApiKey = useCreateApiKey();

  async function handleSubmit(data: ApiKey) {
    await createApiKey.mutateAsync(data);
    navigate({ to: '/apikeys' });
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">{t('apiKeys.title')}</h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate({ to: '/apikeys' })}
          aria-label={t('common.close')}
        >
          <X className="h-5 w-5" />
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl space-y-6">
          <div>
            <h2 className="text-base font-semibold">
              {t('apiKeys.createTitle')}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {t('apiKeys.createDesc')}
            </p>
          </div>

          <ApiKeyForm
            onSubmit={handleSubmit}
            isPending={createApiKey.isPending}
            error={createApiKey.error?.message}
            onCancel={() => navigate({ to: '/apikeys' })}
            submitLabel={t('apiKeys.createApiKey')}
          />
        </div>
      </div>
    </div>
  );
}
