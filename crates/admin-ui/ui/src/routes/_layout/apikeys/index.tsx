import { Link, createFileRoute, useNavigate } from '@tanstack/react-router';
import { createColumnHelper, type ColumnDef } from '@tanstack/react-table';
import { Pencil, Plus, RefreshCw, Trash2 } from 'lucide-react';
import { useState } from 'react';

import { PageHeader } from '@/components/layout/page-header';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { DataTable, type RowSelectionState } from '@/components/ui/data-table';
import { Input } from '@/components/ui/input';
import type { ApiKey, ItemResponse } from '@/lib/api/types';
import {
  useApiKeys,
  useDeleteApiKey,
  useUpdateApiKey,
} from '@/lib/queries/apikeys';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

export const Route = createFileRoute('/_layout/apikeys/')({
  component: ApiKeysPage,
});

type ApiKeyRow = ItemResponse<ApiKey> & { id: string };

const col = createColumnHelper<ApiKeyRow>();

function ApiKeysPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data, isLoading, isError } = useApiKeys();
  const updateApiKey = useUpdateApiKey();
  const deleteApiKey = useDeleteApiKey();

  const [search, setSearch] = useState('');
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  const items = (data?.list ?? []).map(({ key, ...rest }) => ({
    id: key.replace('/apikeys/', ''),
    key,
    ...rest,
  })) as ApiKeyRow[];
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
          onCheckedChange={(v) => table.toggleAllPageRowsSelected(!!v)}
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(v) => row.toggleSelected(!!v)}
        />
      ),
    }),
    col.accessor('id', {
      header: () => t('apiKeys.columns.id'),
      size: 300,
      cell: (info) => (
        <span className="font-mono text-xs text-muted-foreground">
          {info.getValue()}
        </span>
      ),
    }),
    col.accessor('value.key', {
      header: () => t('apiKeys.columns.key'),
      size: 300,
      cell: (info) => (
        <span className="text-sm font-medium">{info.getValue()}</span>
      ),
    }),
    col.accessor('value.allowed_models', {
      header: () => t('apiKeys.columns.allowedModels'),
      cell: (info) => {
        const models = info.getValue() ?? [];
        return models.length === 0 ? (
          <span className="text-xs text-muted-foreground">
            {t('apiKeys.allModels')}
          </span>
        ) : (
          <div className="flex flex-wrap gap-1">
            {models.map((m) => (
              <Badge key={m} variant="secondary" className="font-mono text-xs">
                {m}
              </Badge>
            ))}
          </div>
        );
      },
    }),
    col.display({
      id: 'actions',
      size: 96,
      header: () => (
        <div className="text-right">{t('apiKeys.columns.action')}</div>
      ),
      cell: ({ row }) => (
        <div className="flex justify-end">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-lg"
                onClick={() =>
                  updateApiKey.mutateAsync({
                    id: row.original.id,
                    data: {
                      ...row.original.value,
                      key: 'sk-' + crypto.randomUUID().replaceAll('-', ''),
                    },
                  })
                }
              >
                <RefreshCw />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{t('apiKeys.rotateApiKey')}</p>
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-lg"
                onClick={() =>
                  navigate({
                    to: '/apikeys/$id',
                    params: { id: row.original.id },
                  })
                }
              >
                <Pencil />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{t('apiKeys.editApiKey')}</p>
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="destructive"
                size="icon-lg"
                onClick={() => deleteApiKey.mutate(row.original.id)}
              >
                <Trash2 />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{t('apiKeys.deleteApiKey')}</p>
            </TooltipContent>
          </Tooltip>
        </div>
      ),
    }),
  ];

  async function handleDeleteSelected() {
    for (const id of selectedKeys) {
      await deleteApiKey.mutateAsync(id);
    }
    setRowSelection({});
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">{t('apiKeys.title')}</h1>
        {selectedKeys.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleDeleteSelected}
            disabled={deleteApiKey.isPending}
          >
            <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
            {t('common.delete')} ({selectedKeys.length})
          </Button>
        )}
        <Button asChild size="lg">
          <Link to="/apikeys/create">
            <Plus className="mr-1.5 h-4 w-4" />
            {t('apiKeys.addApiKey')}
          </Link>
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto p-5">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {isLoading
              ? t('common.loading')
              : t('apiKeys.count', { count: items.length })}
          </p>
          <Input
            className="w-80"
            placeholder={t('apiKeys.search')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <DataTable
          columns={columns as ColumnDef<ApiKeyRow>[]}
          data={items}
          isLoading={isLoading}
          isError={isError}
          errorMessage={t('apiKeys.errorLoad')}
          emptyMessage={
            <span>
              {t('apiKeys.empty')}{' '}
              <Link
                to="/apikeys/create"
                className="underline underline-offset-2"
              >
                {t('apiKeys.emptyAction')}
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
