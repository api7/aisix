import { Link, createFileRoute, useNavigate } from '@tanstack/react-router';
import { createColumnHelper, type ColumnDef } from '@tanstack/react-table';
import { Pencil, Plus, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { DataTable, type RowSelectionState } from '@/components/ui/data-table';
import { Input } from '@/components/ui/input';
import type { ItemResponse, Provider } from '@/lib/api/types';
import { useDeleteProvider, useProviders } from '@/lib/queries/providers';

export const Route = createFileRoute('/_layout/providers/')({
  component: ProvidersPage,
});

type ProviderRow = ItemResponse<Provider> & {
  id: string;
  summary: string;
};

const col = createColumnHelper<ProviderRow>();

function ProvidersPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data, isLoading, isError } = useProviders();
  const deleteProvider = useDeleteProvider();

  const [search, setSearch] = useState('');
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  const items = (data?.list ?? []).map(
    ({ key, value, ...rest }) =>
      ({
        id: key.replace('/providers/', ''),
        key,
        value,
        summary: describeProviderConfig(value, t),
        ...rest,
      }) as ProviderRow,
  );
  const selectedKeys = Object.keys(rowSelection);

  const columns = [
    col.display({
      id: 'select',
      size: 40,
      header: ({ table }) => (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && 'indeterminate')
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
        />
      ),
    }),
    col.accessor('id', {
      header: () => t('providers.columns.id'),
      size: 280,
      cell: (info) => (
        <span className="font-mono text-xs text-muted-foreground">
          {info.getValue()}
        </span>
      ),
    }),
    col.accessor('value.name', {
      header: () => t('providers.columns.name'),
      size: 240,
      cell: (info) => (
        <span className="text-sm font-medium">{info.getValue()}</span>
      ),
    }),
    col.accessor('value.type', {
      header: () => t('providers.columns.type'),
      size: 140,
      cell: (info) => (
        <Badge variant="secondary">
          {t(`providers.form.types.${info.getValue()}`)}
        </Badge>
      ),
    }),
    col.accessor('summary', {
      header: () => t('providers.columns.config'),
      cell: (info) => (
        <span className="text-sm text-muted-foreground">{info.getValue()}</span>
      ),
    }),
    col.display({
      id: 'actions',
      size: 96,
      header: () => (
        <div className="text-right">{t('providers.columns.action')}</div>
      ),
      cell: ({ row }) => (
        <div className="flex justify-end">
          <Button
            variant="ghost"
            size="icon-lg"
            onClick={() =>
              navigate({
                to: '/providers/$id',
                params: { id: row.original.id },
              })
            }
          >
            <Pencil />
          </Button>
          <Button
            variant="destructive"
            size="icon-lg"
            onClick={() => deleteProvider.mutate(row.original.id)}
          >
            <Trash2 />
          </Button>
        </div>
      ),
    }),
  ];

  async function handleDeleteSelected() {
    for (const id of selectedKeys) {
      await deleteProvider.mutateAsync(id);
    }
    setRowSelection({});
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">{t('providers.title')}</h1>
        {selectedKeys.length > 0 && (
          <Button
            variant="outline"
            size="lg"
            onClick={handleDeleteSelected}
            disabled={deleteProvider.isPending}
          >
            <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
            {t('common.delete')} ({selectedKeys.length})
          </Button>
        )}
        <Button asChild size="lg">
          <Link to="/providers/create">
            <Plus className="mr-1.5 h-4 w-4" />
            {t('providers.addProvider')}
          </Link>
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto p-5">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {isLoading
              ? t('common.loading')
              : t('providers.count', { count: items.length })}
          </p>
          <Input
            className="w-80"
            placeholder={t('providers.search')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <DataTable
          columns={columns as ColumnDef<ProviderRow>[]}
          data={items}
          isLoading={isLoading}
          isError={isError}
          errorMessage={t('providers.errorLoad')}
          emptyMessage={
            <span>
              {t('providers.empty')}{' '}
              <Link
                to="/providers/create"
                className="underline underline-offset-2"
              >
                {t('providers.emptyAction')}
              </Link>
            </span>
          }
          rowSelection={rowSelection}
          onRowSelectionChange={setRowSelection}
          getRowId={(row) => row.id}
          globalFilter={search}
        />
      </div>
    </div>
  );
}

function describeProviderConfig(
  provider: Provider,
  t: (key: string) => string,
): string {
  if (provider.type === 'bedrock') {
    return provider.config.endpoint
      ? `${provider.config.region} · ${provider.config.endpoint}`
      : provider.config.region;
  }

  return provider.config.api_base ?? t('providers.defaultEndpoint');
}
