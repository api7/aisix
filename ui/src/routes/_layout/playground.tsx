import { createFileRoute } from '@tanstack/react-router';
import { Plus, Send, Trash2 } from 'lucide-react';
import { useRef, useState } from 'react';

import { PageHeader } from '@/components/layout/page-header';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useAdminKey } from '@/hooks/use-admin-key';
import { useModels } from '@/lib/queries/models';

export const Route = createFileRoute('/_layout/playground')({
  component: PlaygroundPage,
});

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface Column {
  id: string;
  modelKey: string;
  messages: Message[];
  isLoading: boolean;
  error?: string;
}

function makeColumn(): Column {
  return {
    id: crypto.randomUUID(),
    modelKey: '',
    messages: [{ role: 'system', content: 'You are a helpful assistant.' }],
    isLoading: false,
  };
}

function PlaygroundPage() {
  const { data: modelsData } = useModels();
  const { key: adminKey } = useAdminKey();
  const models = modelsData?.list ?? [];

  const [columns, setColumns] = useState<Column[]>([
    makeColumn(),
    makeColumn(),
  ]);
  const [userInput, setUserInput] = useState('');

  function updateColumn(id: string, patch: Partial<Column>) {
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

  async function runColumn(col: Column, userMsg: string) {
    if (!col.modelKey || !userMsg.trim()) return;

    const newMessages: Message[] = [
      ...col.messages,
      { role: 'user', content: userMsg },
    ];
    updateColumn(col.id, {
      messages: newMessages,
      isLoading: true,
      error: undefined,
    });

    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(adminKey ? { Authorization: `Bearer ${adminKey}` } : {}),
        },
        body: JSON.stringify({
          model: col.modelKey,
          messages: newMessages.map(({ role, content }) => ({ role, content })),
          stream: false,
        }),
      });

      if (!res.ok) {
        const err = await res
          .json()
          .catch(() => ({ error: { message: res.statusText } }));
        throw new Error(err?.error?.message ?? res.statusText);
      }

      const json = await res.json();
      const assistantContent: string =
        json.choices?.[0]?.message?.content ?? '';
      updateColumn(col.id, {
        messages: [
          ...newMessages,
          { role: 'assistant', content: assistantContent },
        ],
        isLoading: false,
      });
    } catch (e) {
      updateColumn(col.id, {
        messages: newMessages,
        isLoading: false,
        error: String(e instanceof Error ? e.message : e),
      });
    }
  }

  async function handleRun() {
    if (!userInput.trim()) return;
    const msg = userInput;
    setUserInput('');
    await Promise.all(columns.map((col) => runColumn(col, msg)));
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">Playground</h1>
        <Button variant="outline" size="sm" onClick={addColumn}>
          <Plus className="mr-1.5 h-4 w-4" />
          Add Column
        </Button>
      </PageHeader>

      {/* Columns area */}
      <div className="flex flex-1 overflow-hidden">
        {columns.map((col) => (
          <ComparisonColumn
            key={col.id}
            col={col}
            models={models}
            canRemove={columns.length > 1}
            onModelChange={(modelKey) => updateColumn(col.id, { modelKey })}
            onRemove={() => removeColumn(col.id)}
            onClear={() =>
              updateColumn(col.id, {
                messages: [
                  { role: 'system', content: 'You are a helpful assistant.' },
                ],
                error: undefined,
              })
            }
          />
        ))}
      </div>

      {/* Shared input bar */}
      <div className="flex shrink-0 items-end gap-3 border-t bg-background px-4 py-3">
        <Textarea
          className="min-h-[60px] flex-1 resize-none"
          placeholder="Enter a user message and click Run to send to all columns…"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              handleRun();
            }
          }}
        />
        <Button onClick={handleRun} disabled={!userInput.trim()}>
          <Send className="mr-1.5 h-4 w-4" />
          Run
        </Button>
      </div>
    </div>
  );
}

// ── Column component ──────────────────────────────────────────────────────────
interface ComparisonColumnProps {
  col: Column;
  models: Array<{ key: string; value: { name: string; model: string } }>;
  canRemove: boolean;
  onModelChange: (key: string) => void;
  onRemove: () => void;
  onClear: () => void;
}

function ComparisonColumn({
  col,
  models,
  canRemove,
  onModelChange,
  onRemove,
  onClear,
}: ComparisonColumnProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  return (
    <section className="flex min-w-0 flex-1 flex-col border-r last:border-r-0">
      {/* Column header: model selector */}
      <div className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
        <Select value={col.modelKey} onValueChange={onModelChange}>
          <SelectTrigger className="flex-1 text-sm">
            <SelectValue placeholder="Select model…" />
          </SelectTrigger>
          <SelectContent>
            {models.map((m) => (
              <SelectItem key={m.key} value={m.key}>
                <span className="mr-2">{m.value.name}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  {m.value.model}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          variant="ghost"
          size="icon"
          title="Clear messages"
          onClick={onClear}
        >
          <Trash2 className="h-4 w-4 text-muted-foreground" />
        </Button>

        {canRemove && (
          <Button
            variant="ghost"
            size="icon"
            title="Remove column"
            onClick={onRemove}
          >
            <span className="text-muted-foreground">×</span>
          </Button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-auto p-4">
        <div className="space-y-3">
          {col.messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}

          {col.isLoading && (
            <div className="flex justify-start">
              <div className="rounded-lg bg-muted px-3 py-2 text-sm text-muted-foreground">
                Thinking…
              </div>
            </div>
          )}

          {col.error && (
            <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {col.error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>
    </section>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  if (msg.role === 'system') {
    return (
      <div className="flex justify-center">
        <div className="flex items-center gap-1.5 rounded-full border bg-muted/60 px-3 py-1 text-xs text-muted-foreground">
          <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
            system
          </Badge>
          <span className="line-clamp-1">{msg.content}</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={
        msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'
      }
    >
      <div
        className={
          msg.role === 'user'
            ? 'max-w-[85%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground'
            : 'max-w-[85%] rounded-lg bg-muted px-3 py-2 text-sm'
        }
      >
        <pre className="font-sans whitespace-pre-wrap">{msg.content}</pre>
      </div>
    </div>
  );
}
