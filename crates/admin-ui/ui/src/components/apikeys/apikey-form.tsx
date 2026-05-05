import { useForm } from '@tanstack/react-form';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { ApiKey } from '@/lib/api/types';
import { useModels } from '@/lib/queries/models';

export interface ApiKeyFormProps {
  initial?: ApiKey;
  onSubmit: (data: ApiKey) => void | Promise<void>;
  onCancel: () => void;
  isPending: boolean;
  error?: string;
  submitLabel: string;
  extraActions?: React.ReactNode;
}

const RATE_LIMIT_FIELDS = [
  {
    name: 'tpm' as const,
    labelKey: 'apiKeys.form.tpm',
    hintKey: 'apiKeys.form.tpmHint',
  },
  {
    name: 'tpd' as const,
    labelKey: 'apiKeys.form.tpd',
    hintKey: 'apiKeys.form.tpdHint',
  },
  {
    name: 'rpm' as const,
    labelKey: 'apiKeys.form.rpm',
    hintKey: 'apiKeys.form.rpmHint',
  },
  {
    name: 'rpd' as const,
    labelKey: 'apiKeys.form.rpd',
    hintKey: 'apiKeys.form.rpdHint',
  },
  {
    name: 'concurrency' as const,
    labelKey: 'apiKeys.form.concurrency',
    hintKey: undefined,
  },
];

export function ApiKeyForm({
  initial,
  onSubmit,
  onCancel,
  isPending,
  error,
  submitLabel,
  extraActions,
}: ApiKeyFormProps) {
  const { t } = useTranslation();
  const { data: modelsData } = useModels();
  const modelOptions = modelsData?.list ?? [];

  const [allowedModels, setAllowedModels] = useState<string[]>(
    initial?.allowed_models ?? [],
  );
  const [modelInput, setModelInput] = useState('');

  const form = useForm({
    defaultValues: {
      key: initial?.key ?? '',
      tpm:
        initial?.rate_limit?.tpm != null ? String(initial.rate_limit.tpm) : '',
      tpd:
        initial?.rate_limit?.tpd != null ? String(initial.rate_limit.tpd) : '',
      rpm:
        initial?.rate_limit?.rpm != null ? String(initial.rate_limit.rpm) : '',
      rpd:
        initial?.rate_limit?.rpd != null ? String(initial.rate_limit.rpd) : '',
      concurrency:
        initial?.rate_limit?.concurrency != null
          ? String(initial.rate_limit.concurrency)
          : '',
    },
    onSubmit: async ({ value }) => {
      const rateLimit = {
        ...(value.tpm ? { tpm: Number(value.tpm) } : {}),
        ...(value.tpd ? { tpd: Number(value.tpd) } : {}),
        ...(value.rpm ? { rpm: Number(value.rpm) } : {}),
        ...(value.rpd ? { rpd: Number(value.rpd) } : {}),
        ...(value.concurrency
          ? { concurrency: Number(value.concurrency) }
          : {}),
      };
      const payload: ApiKey = {
        key: value.key.trim(),
        allowed_models: allowedModels,
        ...(Object.keys(rateLimit).length > 0 ? { rate_limit: rateLimit } : {}),
      };
      await onSubmit(payload);
    },
  });

  function addModel(m: string) {
    const trimmed = m.trim();
    if (trimmed && !allowedModels.includes(trimmed)) {
      setAllowedModels((prev) => [...prev, trimmed]);
    }
    setModelInput('');
  }

  function removeModel(m: string) {
    setAllowedModels((prev) => prev.filter((x) => x !== m));
  }

  const suggestions = modelOptions.filter(
    (m) => !allowedModels.includes(m.value.name) && m.key.includes(modelInput),
  );

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        e.stopPropagation();
        form.handleSubmit();
      }}
      className="space-y-5"
    >
      {/* Basic */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold">{t('apiKeys.form.basicInfo')}</h3>

        <form.Field name="key">
          {(field) => (
            <Field label={t('apiKeys.form.apiKeyLabel')}>
              <Input
                required
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                onBlur={field.handleBlur}
                placeholder="sk-…"
                autoComplete="off"
              />
              <p className="text-xs text-muted-foreground">
                {t('apiKeys.form.apiKeyHint')}
              </p>
            </Field>
          )}
        </form.Field>
      </section>

      {/* Allowed Models */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold">
          {t('apiKeys.form.allowedModels')}
        </h3>
        <p className="text-xs text-muted-foreground">
          {t('apiKeys.form.allowedModelsHint')}
        </p>

        <div className="flex gap-2">
          <Input
            value={modelInput}
            onChange={(e) => setModelInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addModel(modelInput);
              }
            }}
            placeholder={t('apiKeys.form.modelInputPlaceholder')}
            list="model-suggestions"
          />
          <datalist id="model-suggestions">
            {suggestions.map((s) => (
              <option key={s.value.name} value={s.value.name} />
            ))}
          </datalist>
          <Button
            type="button"
            variant="outline"
            onClick={() => addModel(modelInput)}
          >
            {t('common.add')}
          </Button>
        </div>

        {allowedModels.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {allowedModels.map((m) => (
              <Badge key={m} variant="secondary" asChild>
                <button
                  type="button"
                  className="cursor-pointer font-mono text-xs"
                  onClick={() => removeModel(m)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      removeModel(m);
                    }
                  }}
                  aria-label={`Remove model ${m}`}
                  title="Click to remove"
                >
                  {m} ×
                </button>
              </Badge>
            ))}
          </div>
        )}
      </section>

      {/* Rate limits */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold text-muted-foreground">
          {t('apiKeys.form.rateLimits')}
        </h3>

        <div className="grid grid-cols-3 gap-3">
          {RATE_LIMIT_FIELDS.map(({ name, labelKey, hintKey }) => (
            <form.Field key={name} name={name}>
              {(field) => (
                <Field
                  label={t(labelKey)}
                  hint={hintKey ? t(hintKey) : undefined}
                >
                  <Input
                    type="number"
                    min={0}
                    value={field.state.value}
                    onChange={(e) => field.handleChange(e.target.value)}
                    onBlur={field.handleBlur}
                    placeholder="—"
                  />
                </Field>
              )}
            </form.Field>
          ))}
        </div>
      </section>

      {error && (
        <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between">
        {extraActions ?? <span />}
        <div className="flex gap-2">
          <Button type="button" variant="outline" size="lg" onClick={onCancel}>
            {t('common.cancel')}
          </Button>
          <form.Subscribe selector={(s) => s.isSubmitting}>
            {(isSubmitting) => (
              <Button
                type="submit"
                size="lg"
                disabled={isSubmitting || isPending}
              >
                {isSubmitting || isPending ? t('common.saving') : submitLabel}
              </Button>
            )}
          </form.Subscribe>
        </div>
      </div>
    </form>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-medium">{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
