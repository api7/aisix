import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { Trash2, X } from 'lucide-react';

import { PageHeader } from '@/components/layout/page-header';
import { Button } from '@/components/ui/button';
import { useDeleteModel, useModel, useUpdateModel } from '@/lib/queries/models';
import { ModelForm } from './create';

export const Route = createFileRoute('/_layout/models/$id')({
  component: ModelEditPage,
});

function ModelEditPage() {
  const { id } = Route.useParams();
  const navigate = useNavigate();

  const { data, isLoading, isError } = useModel(id);
  const updateModel = useUpdateModel(id);
  const deleteModel = useDeleteModel();

  async function handleDelete() {
    if (!confirm(`Delete model "${id}"?`)) return;
    await deleteModel.mutateAsync(id);
    navigate({ to: '/models' });
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Loading…
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-destructive">
        Failed to load model.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">Model</h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate({ to: '/models' })}
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl space-y-6">
          <div>
            <h2 className="text-base font-semibold">Edit model resource</h2>
            <p className="mt-1 font-mono text-xs text-muted-foreground">{id}</p>
          </div>

          <ModelForm
            initial={data.value}
            onSubmit={async (payload) => {
              await updateModel.mutateAsync(payload);
              navigate({ to: '/models' });
            }}
            isPending={updateModel.isPending}
            error={updateModel.error?.message}
            onCancel={() => navigate({ to: '/models' })}
            submitLabel="Save Changes"
            extraActions={
              <Button
                type="button"
                variant="outline"
                onClick={handleDelete}
                disabled={deleteModel.isPending}
              >
                <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
                Delete Model
              </Button>
            }
          />
        </div>
      </div>
    </div>
  );
}
