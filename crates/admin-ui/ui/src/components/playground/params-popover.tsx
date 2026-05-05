import { WandSparkles } from 'lucide-react';
import {
  Suspense,
  lazy,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from 'react';
import { useTranslation } from 'react-i18next';

import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';

import type { ColumnState, Params } from './types';

interface ParamsPopoverProps {
  params: Params;
  onPatch: (patch: Partial<ColumnState>) => void;
  onClose: () => void;
  anchorRef: RefObject<HTMLElement | null>;
}

const POPOVER_WIDTH = 460;
const MonacoJsonEditor = lazy(async () => {
  const mod = await import('@/components/ui/monaco-json-editor');
  return { default: mod.MonacoJsonEditor };
});

export function ParamsPopover({
  params,
  onPatch,
  onClose,
  anchorRef,
}: ParamsPopoverProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);

  useLayoutEffect(() => {
    function updatePosition() {
      const anchor = anchorRef.current;
      if (!anchor) return;

      const rect = anchor.getBoundingClientRect();
      // right-align the popover to the button's right edge
      const right = rect.right;
      const left = Math.max(8, right - POPOVER_WIDTH);
      setPosition({ left, top: rect.bottom + 8 });
    }

    updatePosition();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
    };
  }, [anchorRef]);

  useEffect(() => {
    function onDown(e: MouseEvent) {
      const target = e.target as Node;
      if (ref.current?.contains(target)) return;
      if (anchorRef.current?.contains(target)) return;
      onClose();
    }
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [anchorRef, onClose]);

  return (
    <div
      ref={ref}
      className="fixed z-50 w-115 overflow-hidden rounded-xl border bg-card shadow-[0_8px_30px_rgba(0,0,0,0.15)]"
      style={{
        left: position?.left ?? 8,
        top: position?.top ?? 8,
        visibility: position ? 'visible' : 'hidden',
      }}
    >
      <div className="flex h-11.5 items-center justify-between border-b px-4">
        <span className="text-[13px] font-semibold">
          {t('playground.params.title')}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold tracking-widest">
            {t('playground.custom')}
          </span>
          <button
            role="switch"
            aria-checked={params.custom}
            onClick={() => {
              const nextCustom = !params.custom;

              // Keep stream config inside JSON when entering custom mode.
              if (nextCustom) {
                try {
                  const parsed = JSON.parse(params.json || '{}');
                  const nextJson = JSON.stringify(
                    { ...parsed, stream: params.stream },
                    null,
                    2,
                  );
                  onPatch({
                    params: { ...params, custom: true, json: nextJson },
                  });
                  return;
                } catch {
                  const nextJson = JSON.stringify(
                    { stream: params.stream },
                    null,
                    2,
                  );
                  onPatch({
                    params: { ...params, custom: true, json: nextJson },
                  });
                  return;
                }
              }

              // Restore stream switch value from custom JSON when leaving custom mode.
              try {
                const parsed = JSON.parse(params.json || '{}') as {
                  stream?: unknown;
                };
                const nextStream =
                  typeof parsed.stream === 'boolean'
                    ? parsed.stream
                    : params.stream;
                onPatch({
                  params: { ...params, custom: false, stream: nextStream },
                });
              } catch {
                onPatch({ params: { ...params, custom: false } });
              }
            }}
            className={cn(
              'relative inline-flex h-5 w-9 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              params.custom ? 'bg-primary' : 'bg-input',
            )}
          >
            <span
              className={cn(
                'block h-4 w-4 rounded-full bg-white shadow-md transition-transform',
                params.custom ? 'translate-x-4' : 'translate-x-0',
              )}
            />
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-3 p-4">
        {!params.custom && (
          <div className="flex items-center justify-between rounded-md border bg-background px-3 py-2">
            <span className="text-xs font-medium">
              {t('playground.params.stream')}
            </span>
            <button
              role="switch"
              aria-checked={params.stream}
              onClick={() =>
                onPatch({ params: { ...params, stream: !params.stream } })
              }
              className={cn(
                'relative inline-flex h-5 w-9 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                params.stream ? 'bg-primary' : 'bg-input',
              )}
            >
              <span
                className={cn(
                  'block h-4 w-4 rounded-full bg-white shadow-md transition-transform',
                  params.stream ? 'translate-x-4' : 'translate-x-0',
                )}
              />
            </button>
          </div>
        )}

        {params.custom ? (
          <div className="overflow-hidden rounded-lg border bg-background">
            <div className="flex h-8.5 items-center justify-between border-b px-2.5">
              <span className="font-mono text-[11px] font-medium text-muted-foreground">
                {t('playground.customJson')}
              </span>
              <button
                className="text-muted-foreground hover:text-foreground"
                title={t('playground.formatJson')}
                aria-label={t('playground.formatJson')}
                onClick={() => {
                  try {
                    onPatch({
                      params: {
                        ...params,
                        json: JSON.stringify(JSON.parse(params.json), null, 2),
                      },
                    });
                  } catch {
                    // invalid json — do nothing
                  }
                }}
              >
                <WandSparkles className="h-3.5 w-3.5" />
              </button>
            </div>
            <Suspense
              fallback={
                <div className="h-[246px] rounded-none border-0 bg-muted/30" />
              }
            >
              <MonacoJsonEditor
                className="rounded-none border-0"
                height={246}
                value={params.json}
                onChange={(next) =>
                  onPatch({ params: { ...params, json: next } })
                }
                ariaLabel="Custom JSON parameters"
              />
            </Suspense>
          </div>
        ) : (
          <>
            <div className="flex gap-3">
              <div className="flex-1">
                <Label className="mb-1.5 block text-xs">
                  {t('playground.params.maxTokens')}
                </Label>
                <Input
                  type="number"
                  className="h-8 text-sm"
                  value={params.max_tokens}
                  onChange={(e) =>
                    onPatch({
                      params: { ...params, max_tokens: e.target.value },
                    })
                  }
                />
              </div>
              <div className="flex-1">
                <Label className="mb-1.5 block text-xs">
                  {t('playground.params.temperature')}
                </Label>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  className="h-8 text-sm"
                  value={params.temperature}
                  onChange={(e) =>
                    onPatch({
                      params: { ...params, temperature: e.target.value },
                    })
                  }
                />
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex-1">
                <Label className="mb-1.5 block text-xs">
                  {t('playground.params.topP')}
                </Label>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  className="h-8 text-sm"
                  value={params.top_p}
                  onChange={(e) =>
                    onPatch({ params: { ...params, top_p: e.target.value } })
                  }
                />
              </div>
              <div className="flex-1">
                <Label className="mb-1.5 block text-xs">
                  {t('playground.params.n')}
                </Label>
                <Input
                  type="number"
                  min="1"
                  className="h-8 text-sm"
                  value={params.n}
                  onChange={(e) =>
                    onPatch({ params: { ...params, n: e.target.value } })
                  }
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
