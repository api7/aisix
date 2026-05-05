import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { Trash2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';
import { ProviderForm } from '@/components/providers/provider-form';
import { Button } from '@/components/ui/button';
import {
  useDeleteProvider,
  useProvider,
  useUpdateProvider,
} from '@/lib/queries/providers';

export const Route = createFileRoute('/_layout/providers/$id')({
  component: ProviderEditPage,
});

function ProviderEditPage() {
  const { t } = useTranslation();
  const { id } = Route.useParams();
  const navigate = useNavigate();

  const { data, isLoading, isError } = useProvider(id);
  const updateProvider = useUpdateProvider(id);
  const deleteProvider = useDeleteProvider();

  async function handleDelete() {
    if (!confirm(t('providers.deleteConfirm', { id }))) return;
    await deleteProvider.mutateAsync(id);
    navigate({ to: '/providers' });
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
        {t('providers.errorLoadSingle')}
      </div>
    );
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
              {t('providers.editTitle')}
            </h2>
            <p className="mt-1 font-mono text-xs text-muted-foreground">{id}</p>
          </div>

          <ProviderForm
            initial={data.value}
            onSubmit={async (payload) => {
              await updateProvider.mutateAsync(payload);
              navigate({ to: '/providers' });
            }}
            isPending={updateProvider.isPending}
            error={updateProvider.error?.message}
            onCancel={() => navigate({ to: '/providers' })}
            submitLabel={t('common.saveChanges')}
            extraActions={
              <Button
                type="button"
                variant="outline"
                onClick={handleDelete}
                disabled={deleteProvider.isPending}
              >
                <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
                {t('providers.deleteProvider')}
              </Button>
            }
          />
        </div>
      </div>
    </div>
  );
}
