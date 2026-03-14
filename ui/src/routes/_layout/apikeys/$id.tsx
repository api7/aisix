import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { Trash2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';
import { Button } from '@/components/ui/button';
import {
  useApiKey,
  useDeleteApiKey,
  useUpdateApiKey,
} from '@/lib/queries/apikeys';
import { ApiKeyForm } from '@/components/apikeys/apikey-form';

export const Route = createFileRoute('/_layout/apikeys/$id')({
  component: ApiKeyEditPage,
});

function ApiKeyEditPage() {
  const { t } = useTranslation();
  const { id } = Route.useParams();
  const navigate = useNavigate();

  const { data, isLoading, isError } = useApiKey(id);
  const updateApiKey = useUpdateApiKey();
  const deleteApiKey = useDeleteApiKey();

  async function handleDelete() {
    if (!confirm(t('apiKeys.deleteConfirm', { id }))) return;
    await deleteApiKey.mutateAsync(id);
    navigate({ to: '/apikeys' });
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        {t('common.loading')}
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-destructive">
        {t('apiKeys.errorLoadSingle')}
      </div>
    );
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
              {t('apiKeys.editTitle')}
            </h2>
            <p className="mt-1 font-mono text-xs text-muted-foreground">{id}</p>
          </div>

          <ApiKeyForm
            initial={data.value}
            onSubmit={async (payload) => {
              await updateApiKey.mutateAsync({ id, data: payload });
              navigate({ to: '/apikeys' });
            }}
            isPending={updateApiKey.isPending}
            error={updateApiKey.error?.message}
            onCancel={() => navigate({ to: '/apikeys' })}
            submitLabel={t('common.saveChanges')}
            extraActions={
              <Button
                type="button"
                variant="outline"
                size="lg"
                onClick={handleDelete}
                disabled={deleteApiKey.isPending}
              >
                <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
                {t('apiKeys.deleteApiKey')}
              </Button>
            }
          />
        </div>
      </div>
    </div>
  );
}
