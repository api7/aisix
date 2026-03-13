import { Link, createFileRoute, useNavigate } from '@tanstack/react-router';
import { createColumnHelper, type ColumnDef } from '@tanstack/react-table';
import { Pencil, Plus, Trash2 } from 'lucide-react';
import { useState } from 'react';

import { PageHeader } from '@/components/layout/page-header';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { DataTable, type RowSelectionState } from '@/components/ui/data-table';
import { Input } from '@/components/ui/input';
import type { ItemResponse, Model } from '@/lib/api/types';
import { useDeleteModel, useModels } from '@/lib/queries/models';

export const Route = createFileRoute('/_layout/models/')({
  component: ModelsPage,
});

type ModelRow = ItemResponse<Model> & { id: string };

const col = createColumnHelper<ModelRow>();

function ModelsPage() {
  const navigate = useNavigate();
  const { data, isLoading, isError } = useModels();
  const deleteModel = useDeleteModel();

  const [search, setSearch] = useState('');
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  const items = (data?.list ?? []).map(
    ({ key, ...rest }) =>
      ({
        id: key.replace('/models/', ''),
        key,
        ...rest,
      }) as ModelRow,
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
    col.accessor('value.name', {
      header: 'Name',
      size: 300,
      cell: (info) => (
        <span className="text-sm font-medium">{info.getValue()}</span>
      ),
    }),
    col.accessor('value.model', {
      header: 'Model',
      cell: (info) => (
        <Badge variant="secondary" className="font-mono text-xs">
          {info.getValue()}
        </Badge>
      ),
    }),
    col.display({
      id: 'actions',
      size: 96,
      header: () => <div className="text-right">Action</div>,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <Button
            variant="ghost"
            size="icon-lg"
            onClick={() =>
              navigate({
                to: '/models/$id',
                params: { id: row.original.id },
              })
            }
          >
            <Pencil />
          </Button>
          <Button
            variant="destructive"
            size="icon-lg"
            onClick={() => deleteModel.mutate(row.original.id)}
          >
            <Trash2 />
          </Button>
        </div>
      ),
    }),
  ];

  async function handleDeleteSelected() {
    for (const id of selectedKeys) {
      await deleteModel.mutateAsync(id);
    }
    setRowSelection({});
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">Models</h1>
        {selectedKeys.length > 0 && (
          <Button
            variant="outline"
            size="lg"
            onClick={handleDeleteSelected}
            disabled={deleteModel.isPending}
          >
            <Trash2 className="mr-1.5 h-4 w-4 text-destructive" />
            Delete ({selectedKeys.length})
          </Button>
        )}
        <Button asChild size="lg">
          <Link to="/models/create">
            <Plus className="mr-1.5 h-4 w-4" />
            Add Model
          </Link>
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto p-5">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {isLoading
              ? 'Loading…'
              : `${items.length} model${items.length !== 1 ? 's' : ''}`}
          </p>
          <Input
            className="w-80"
            placeholder="Search models…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <DataTable
          columns={columns as ColumnDef<ModelRow>[]}
          data={items}
          isLoading={isLoading}
          isError={isError}
          errorMessage="Failed to load models."
          emptyMessage={
            <span>
              No models yet.{' '}
              <Link
                to="/models/create"
                className="underline underline-offset-2"
              >
                Add one
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
