import { ChevronDown, X } from 'lucide-react';
import { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';

import { ROLES, type Message, type Role } from './types';

const MonacoJsonEditor = lazy(async () => {
  const mod = await import('@/components/ui/monaco-json-editor');
  return { default: mod.MonacoJsonEditor };
});

interface MsgCardProps {
  msg: Message;
  onChange: (patch: Partial<Message>) => void;
  onDelete?: () => void;
}

export function MsgCard({ msg, onChange, onDelete }: MsgCardProps) {
  const { t } = useTranslation();
  const [roleOpen, setRoleOpen] = useState(false);
  const roleRef = useRef<HTMLDivElement>(null);

  const isJson = msg.role === 'json';

  useEffect(() => {
    if (!roleOpen) return;
    function onDown(e: MouseEvent) {
      if (!roleRef.current?.contains(e.target as Node)) setRoleOpen(false);
    }
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [roleOpen]);

  return (
    <div className="relative rounded-lg border bg-card">
      {/* Header */}
      <div className="flex h-10 items-center justify-between border-b px-3">
        <div className="relative" ref={roleRef}>
          <button
            className="flex items-center gap-1.5 rounded border border-border bg-background px-2 py-1 text-xs font-medium text-foreground"
            onClick={() => setRoleOpen((o) => !o)}
          >
            {t(`playground.roles.${msg.role}`)}
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </button>

          {roleOpen && (
            <div className="absolute top-full left-0 z-80 mt-1 w-40 overflow-hidden rounded-lg border bg-card shadow-[0_6px_24px_rgba(0,0,0,0.1)]">
              {ROLES.map((r) => (
                <button
                  key={r}
                  className={cn(
                    'flex h-8.5 w-full items-center px-3 text-xs font-medium hover:bg-muted',
                    r === msg.role && 'bg-muted',
                  )}
                  onClick={() => {
                    if (r === 'json' && msg.role !== 'json') {
                      onChange({
                        role: 'json',
                        content: JSON.stringify(
                          { role: msg.role, content: msg.content },
                          null,
                          2,
                        ),
                      });
                    } else if (msg.role === 'json' && r !== 'json') {
                      let nextContent = msg.content;
                      try {
                        const parsed = JSON.parse(msg.content);
                        if (
                          parsed &&
                          typeof parsed === 'object' &&
                          typeof parsed.content === 'string'
                        ) {
                          nextContent = parsed.content;
                        }
                      } catch {
                        // Keep original content when JSON is invalid.
                      }
                      onChange({ role: r, content: nextContent });
                    } else {
                      onChange({ role: r });
                    }
                    setRoleOpen(false);
                  }}
                >
                  {t(`playground.roles.${r}`)}
                </button>
              ))}
            </div>
          )}
        </div>

        {onDelete ? (
          <button
            className="text-muted-foreground hover:text-foreground"
            title={t('playground.deleteMessage')}
            onClick={onDelete}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        ) : null}
      </div>

      {/* Body */}
      <div className="p-3.5">
        {isJson ? (
          <Suspense
            fallback={
              <div className="h-[220px] rounded-md border bg-muted/30" />
            }
          >
            <MonacoJsonEditor
              value={msg.content}
              height={220}
              onChange={(next) => onChange({ content: next })}
              ariaLabel="JSON message editor"
            />
          </Suspense>
        ) : (
          <Textarea
            className="min-h-16 resize-none rounded-md border border-border/70 bg-background/60 px-3 py-2 text-[13px] leading-relaxed shadow-none focus-visible:ring-1"
            placeholder={
              msg.role === 'user'
                ? t('playground.msgPlaceholder')
                : msg.role === 'assistant'
                  ? t('playground.assistantPlaceholder')
                  : t('playground.systemPlaceholder', 'System instructions…')
            }
            value={msg.content}
            onChange={(e) => onChange({ content: e.target.value })}
          />
        )}
      </div>
    </div>
  );
}

export function RoleBadge({ role }: { role: Role }) {
  const { t } = useTranslation();
  return (
    <span className="rounded border border-border bg-background px-2 py-1 text-xs font-medium text-foreground">
      {t(`playground.roles.${role}`)}
    </span>
  );
}
