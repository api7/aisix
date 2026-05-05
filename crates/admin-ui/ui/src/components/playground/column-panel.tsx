import { Play, Plus, RefreshCw, SlidersHorizontal, Trash2 } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';

import { ApiKeyPicker } from './api-key-picker';
import { MsgCard, RoleBadge } from './msg-card';
import { ModelPicker } from './model-picker';
import { ParamsPopover } from './params-popover';
import {
  makeMsgId,
  type ApiKeyItem,
  type ColumnState,
  type Message,
  type ModelItem,
} from './types';

interface ColumnPanelProps {
  col: ColumnState;
  apiKeys: ApiKeyItem[];
  models: ModelItem[];
  canRemove: boolean;
  isRefreshingModels?: boolean;
  onPatch: (patch: Partial<ColumnState>) => void;
  onRemove: () => void;
  onRefreshModels: () => void;
  onRun: () => void;
}

export function ColumnPanel({
  col,
  apiKeys,
  models,
  canRemove,
  isRefreshingModels,
  onPatch,
  onRemove,
  onRefreshModels,
  onRun,
}: ColumnPanelProps) {
  const { t } = useTranslation();
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const paramsButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const node = messagesScrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' });
  }, [col.messages.length, col.isLoading]);

  const selected = models.find((m) => m.key === col.modelKey);
  const modelLabel = selected
    ? selected.value.name
    : t('playground.selectModel');

  function patchMsg(id: string, patch: Partial<Message>) {
    onPatch({
      messages: col.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    });
  }

  function deleteMsg(id: string) {
    onPatch({ messages: col.messages.filter((m) => m.id !== id) });
  }

  function addMsg() {
    onPatch({
      messages: [
        ...col.messages,
        { id: makeMsgId(), role: 'user', content: '' },
      ],
    });
  }

  return (
    <div className="relative flex h-full min-w-100 flex-1 basis-100 flex-col border-r last:border-r-0">
      {/* ── Header ── */}
      <div className="flex h-14 shrink-0 items-center justify-between border-b bg-background px-4">
        <div className="flex items-center">
          <ApiKeyPicker
            apiKeys={apiKeys}
            value={col.apiKeyKey}
            onChange={(v) => onPatch({ apiKeyKey: v })}
          />
          <div className="ml-1.5 flex items-center">
            <ModelPicker
              label={modelLabel}
              models={models}
              value={col.modelKey}
              buttonClassName="rounded-r-none border-r-0"
              onChange={(v) => onPatch({ modelKey: v })}
            />
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="h-8 w-8 rounded-l-none"
              title={t('playground.refreshModels')}
              aria-label={t('playground.refreshModels')}
              onClick={onRefreshModels}
              disabled={!!isRefreshingModels}
            >
              <RefreshCw
                className={
                  isRefreshingModels ? 'h-4 w-4 animate-spin' : 'h-4 w-4'
                }
              />
            </Button>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            ref={paramsButtonRef}
            variant="ghost"
            size="sm"
            className="h-8 gap-1.5 text-[13px]"
            onClick={() => onPatch({ parametersOpen: !col.parametersOpen })}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            {t('playground.parameters')}
          </Button>
          {canRemove && (
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={onRemove}
            >
              <Trash2 className="h-4 w-4 text-destructive" />
            </Button>
          )}
        </div>
      </div>

      {/* ── Parameters popover ── */}
      {col.parametersOpen && (
        <ParamsPopover
          params={col.params}
          onPatch={onPatch}
          onClose={() => onPatch({ parametersOpen: false })}
          anchorRef={paramsButtonRef}
        />
      )}

      {/* ── Messages ── */}
      <div
        ref={messagesScrollRef}
        className="flex-1 overflow-y-auto bg-muted p-4"
      >
        <div className="flex flex-col gap-3">
          {col.messages.map((msg) => (
            <MsgCard
              key={msg.id}
              msg={msg}
              onChange={(patch) => patchMsg(msg.id, patch)}
              onDelete={() => deleteMsg(msg.id)}
            />
          ))}

          {col.isLoading && !col.params.stream && (
            <div className="overflow-hidden rounded-lg border bg-card">
              <div className="flex h-10 items-center border-b px-3">
                <RoleBadge role="assistant" />
              </div>
              <div className="p-3 text-[13px] text-muted-foreground">
                {t('playground.generating')}
              </div>
            </div>
          )}

          {col.error && (
            <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {col.error}
            </div>
          )}
        </div>
      </div>

      {/* ── Footer ── */}
      <div className="flex h-14 shrink-0 items-center justify-between border-t bg-background px-4">
        <Button
          type="button"
          variant="ghost"
          size="lg"
          className="gap-1.5"
          onClick={addMsg}
        >
          <Plus className="h-4 w-4" />
          {t('playground.addMessage')}
        </Button>
        <Button
          type="button"
          size="lg"
          className="gap-1.5"
          disabled={col.isLoading || !col.modelKey || !col.apiKeyKey}
          onClick={onRun}
        >
          <Play className="h-4 w-4" />
          {t('playground.generateCompletion')}
        </Button>
      </div>
    </div>
  );
}
