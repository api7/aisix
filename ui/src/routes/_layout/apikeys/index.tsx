import { Link, createFileRoute, useNavigate } from '@tanstack/react-router';
import { createColumnHelper, type ColumnDef } from '@tanstack/react-table';
import { Pencil, Plus, RefreshCw, Trash2 } from 'lucide-react';
import { useState } from 'react';

import { PageHeader } from '@/components/layout/page-header';
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
      header: 'ID',
      size: 300,
      cell: (info) => (
        <span className="font-mono text-xs text-muted-foreground">
          {info.getValue()}
        </span>
      ),
    }),
    col.accessor('value.key', {
      header: 'Key',
      size: 300,
      cell: (info) => (
        <span className="text-sm font-medium">{info.getValue()}</span>
      ),
    }),
    col.accessor('value.allowed_models', {
      header: 'Allowed Models',
      cell: (info) => {
        const models = info.getValue() ?? [];
        return models.length === 0 ? (
          <span className="text-xs text-muted-foreground">All models</span>
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
      header: () => <div className="text-right">Action</div>,
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
              <p>Rotate API Key</p>
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
              <p>Edit API Key</p>
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
              <p>Delete API Key</p>
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
        <h1 className="flex-1 text-xl font-semibold">API Keys</h1>
        {selectedKeys.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleDeleteSelected}
            disabled={deleteApiKey.isPending}
          >
            <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
            Delete ({selectedKeys.length})
          </Button>
        )}
        <Button asChild size="lg">
          <Link to="/apikeys/create">
            <Plus className="mr-1.5 h-4 w-4" />
            Add API Key
          </Link>
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto p-5">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {isLoading
              ? 'Loading…'
              : `${items.length} key${items.length !== 1 ? 's' : ''}`}
          </p>
          <Input
            className="w-80"
            placeholder="Search by key…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <DataTable
          columns={columns as ColumnDef<ApiKeyRow>[]}
          data={items}
          isLoading={isLoading}
          isError={isError}
          errorMessage="Failed to load API keys."
          emptyMessage={
            <span>
              No API keys yet.{' '}
              <Link
                to="/apikeys/create"
                className="underline underline-offset-2"
              >
                Create one
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
