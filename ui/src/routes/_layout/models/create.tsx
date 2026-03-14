import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';
import { ModelForm } from '@/components/models/model-form';
import { Button } from '@/components/ui/button';
import type { Model } from '@/lib/api/types';
import { useCreateModel } from '@/lib/queries/models';

export const Route = createFileRoute('/_layout/models/create')({
  component: ModelCreatePage,
});

function ModelCreatePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createModel = useCreateModel();

  async function handleSubmit(data: Model) {
    await createModel.mutateAsync(data);
    navigate({ to: '/models' });
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">{t('models.title')}</h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate({ to: '/models' })}
          aria-label={t('common.close')}
        >
          <X className="h-5 w-5" />
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl space-y-6">
          <div>
            <h2 className="text-base font-semibold">
              {t('models.createTitle')}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {t('models.createDesc')}
            </p>
          </div>

          <ModelForm
            onSubmit={handleSubmit}
            isPending={createModel.isPending}
            error={createModel.error?.message}
            onCancel={() => navigate({ to: '/models' })}
            submitLabel={t('models.createModel')}
          />
        </div>
      </div>
    </div>
  );
}
