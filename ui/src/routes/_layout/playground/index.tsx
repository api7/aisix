import { createFileRoute } from '@tanstack/react-router';
import { Plus, Save } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { PageHeader } from '@/components/layout/page-header';
import { runPlaygroundCompletion } from '@/components/playground/api';
import { ColumnPanel } from '@/components/playground/column-panel';
import {
  parseStoredColumns,
  PLAYGROUND_COLUMNS_STORAGE_KEY,
  toApiMessages,
} from '@/components/playground/state';
import { makeColumn, type ColumnState } from '@/components/playground/types';
import { Button } from '@/components/ui/button';
import { useApiKeys } from '@/lib/queries/apikeys';
import { useModels } from '@/lib/queries/models';

export const Route = createFileRoute('/_layout/playground/')({
  component: PlaygroundPage,
});

function PlaygroundPage() {
  const { t } = useTranslation();
  const { data: modelsData, refetch: refetchModels, isFetching } = useModels();
  const { data: apiKeysData } = useApiKeys();
  const models = modelsData?.list ?? [];
  const apiKeys = apiKeysData?.list ?? [];

  const [columns, setColumns] = useState<ColumnState[]>(() => {
    if (typeof window === 'undefined') return [makeColumn()];
    const stored = parseStoredColumns(
      window.localStorage.getItem(PLAYGROUND_COLUMNS_STORAGE_KEY),
    );
    return stored && stored.length > 0 ? stored : [makeColumn()];
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const snapshot = columns.map((col) => ({
      id: col.id,
      apiKeyKey: col.apiKeyKey,
      modelKey: col.modelKey,
      messages: col.messages,
      params: col.params,
    }));

    window.localStorage.setItem(
      PLAYGROUND_COLUMNS_STORAGE_KEY,
      JSON.stringify(snapshot),
    );
  }, [columns]);

  const canSave = useMemo(
    () => columns.some((c) => c.messages.some((m) => m.content.trim())),
    [columns],
  );

  function patchColumn(id: string, patch: Partial<ColumnState>) {
    setColumns((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...patch } : c)),
    );
  }

  function addColumn() {
    setColumns((prev) => [...prev, makeColumn()]);
  }

  function removeColumn(id: string) {
    setColumns((prev) => prev.filter((c) => c.id !== id));
  }

  async function runColumn(column: ColumnState) {
    if (!column.apiKeyKey) {
      patchColumn(column.id, { error: t('playground.apiKeyRequired') });
      return;
    }
    if (!column.modelKey) return;

    patchColumn(column.id, { isLoading: true, error: undefined });

    try {
      let customBody: Record<string, unknown> | null = null;
      let streamMode = column.params.stream;
      let parseErrorMessage: string | undefined;

      if (column.params.custom) {
        try {
          const parsed = JSON.parse(column.params.json || '{}') as unknown;
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            customBody = parsed as Record<string, unknown>;
            if (typeof customBody.stream === 'boolean') {
              streamMode = customBody.stream;
            }
          } else {
            customBody = null;
          }
        } catch (err) {
          console.error('Invalid custom JSON in playground params:', err);
          customBody = null;
          streamMode = column.params.stream;
          parseErrorMessage = t('playground.invalidJson');
          patchColumn(column.id, { error: parseErrorMessage });
        }
      }

      const paramsBody = column.params.custom
        ? customBody
        : {
            ...(column.params.max_tokens
              ? { max_tokens: Number(column.params.max_tokens) }
              : {}),
            ...(column.params.temperature
              ? { temperature: Number(column.params.temperature) }
              : {}),
            ...(column.params.top_p
              ? { top_p: Number(column.params.top_p) }
              : {}),
            ...(column.params.n ? { n: Number(column.params.n) } : {}),
          };

      const selectedModel = models.find((m) => m.key === column.modelKey);
      const modelName = selectedModel?.value.name ?? column.modelKey;
      const selectedApiKey = apiKeys.find(
        (item) => item.key === column.apiKeyKey,
      );

      if (!selectedApiKey) {
        throw new Error(t('playground.apiKeyRequired'));
      }

      if (streamMode) {
        const assistantId = crypto.randomUUID();
        setColumns((prev) =>
          prev.map((c) =>
            c.id === column.id
              ? {
                  ...c,
                  messages: [
                    ...c.messages,
                    { id: assistantId, role: 'assistant', content: '' },
                  ],
                }
              : c,
          ),
        );

        await runPlaygroundCompletion({
          apiKey: selectedApiKey.value.key,
          model: modelName,
          messages: toApiMessages(column),
          paramsBody,
          stream: true,
          onStreamChunk: (content) => {
            setColumns((prev) =>
              prev.map((c) =>
                c.id === column.id
                  ? {
                      ...c,
                      messages: c.messages.map((m) =>
                        m.id === assistantId ? { ...m, content } : m,
                      ),
                    }
                  : c,
              ),
            );
          },
        });

        patchColumn(column.id, {
          isLoading: false,
          error: parseErrorMessage,
        });
      } else {
        const content = await runPlaygroundCompletion({
          apiKey: selectedApiKey.value.key,
          model: modelName,
          messages: toApiMessages(column),
          paramsBody,
          stream: false,
        });

        setColumns((prev) =>
          prev.map((c) =>
            c.id === column.id
              ? {
                  ...c,
                  isLoading: false,
                  error: parseErrorMessage,
                  messages: [
                    ...c.messages,
                    {
                      id: crypto.randomUUID(),
                      role: 'assistant',
                      content,
                    },
                  ],
                }
              : c,
          ),
        );
      }
    } catch (e) {
      patchColumn(column.id, {
        isLoading: false,
        error: String(e instanceof Error ? e.message : e),
      });
    }
  }

  function savePrompt() {
    const payload = {
      created_at: new Date().toISOString(),
      columns: columns.map((c) => ({
        modelKey: c.modelKey,
        params: c.params,
        messages: c.messages,
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `playground-prompt-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex h-full flex-col bg-background">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">
          {t('playground.title')}
        </h1>
        <Button
          type="button"
          variant="outline"
          size="lg"
          className="gap-1.5"
          onClick={addColumn}
        >
          <Plus className="h-4 w-4" />
          {t('playground.addComparison')}
        </Button>
        <Button
          type="button"
          variant="default"
          size="lg"
          className="gap-1.5"
          onClick={savePrompt}
          disabled={!canSave}
        >
          <Save className="h-4 w-4" />
          {t('playground.savePrompt')}
        </Button>
      </PageHeader>

      <div className="min-h-0 flex-1 overflow-x-auto overflow-y-hidden">
        <div className="flex h-full w-fit min-w-full">
          {columns.map((col) => (
            <ColumnPanel
              key={col.id}
              col={col}
              apiKeys={apiKeys}
              models={models}
              canRemove={columns.length > 1}
              isRefreshingModels={isFetching}
              onPatch={(patch) => patchColumn(col.id, patch)}
              onRemove={() => removeColumn(col.id)}
              onRefreshModels={() => {
                void refetchModels();
              }}
              onRun={() => runColumn(col)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
