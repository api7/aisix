import { useForm } from '@tanstack/react-form';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { Model } from '@/lib/api/types';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ModelFormProps {
  initial?: Model;
  onSubmit: (data: Model) => void | Promise<void>;
  onCancel: () => void;
  isPending: boolean;
  error?: string;
  submitLabel: string;
  extraActions?: React.ReactNode;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const RATE_LIMIT_FIELDS = [
  {
    name: 'tpm' as const,
    labelKey: 'models.form.tpm',
    hintKey: 'models.form.tpmHint',
  },
  {
    name: 'tpd' as const,
    labelKey: 'models.form.tpd',
    hintKey: 'models.form.tpdHint',
  },
  {
    name: 'rpm' as const,
    labelKey: 'models.form.rpm',
    hintKey: 'models.form.rpmHint',
  },
  {
    name: 'rpd' as const,
    labelKey: 'models.form.rpd',
    hintKey: 'models.form.rpdHint',
  },
  {
    name: 'concurrency' as const,
    labelKey: 'models.form.concurrency',
    hintKey: undefined,
  },
];

function parseOptionalNonNegativeInteger(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }

  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
    return undefined;
  }

  return parsed;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function ModelForm({
  initial,
  onSubmit,
  onCancel,
  isPending,
  error,
  submitLabel,
  extraActions,
}: ModelFormProps) {
  const { t } = useTranslation();
  const form = useForm({
    defaultValues: {
      name: initial?.name ?? '',
      model: initial?.model ?? '',
      api_key: initial?.provider_config.api_key ?? '',
      api_base: initial?.provider_config.api_base ?? '',
      timeout: initial?.timeout != null ? String(initial.timeout) : '',
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
      const tpm = parseOptionalNonNegativeInteger(value.tpm);
      const tpd = parseOptionalNonNegativeInteger(value.tpd);
      const rpm = parseOptionalNonNegativeInteger(value.rpm);
      const rpd = parseOptionalNonNegativeInteger(value.rpd);
      const concurrency = parseOptionalNonNegativeInteger(value.concurrency);
      const timeout = parseOptionalNonNegativeInteger(value.timeout);

      const rateLimit: NonNullable<Model['rate_limit']> = {
        ...(tpm != null ? { tpm } : {}),
        ...(tpd != null ? { tpd } : {}),
        ...(rpm != null ? { rpm } : {}),
        ...(rpd != null ? { rpd } : {}),
        ...(concurrency != null ? { concurrency } : {}),
      };
      const payload: Model = {
        name: value.name.trim(),
        model: value.model.trim(),
        provider_config: {
          ...(value.api_key ? { api_key: value.api_key.trim() } : {}),
          ...(value.api_base ? { api_base: value.api_base.trim() } : {}),
        },
        ...(timeout != null ? { timeout } : {}),
        ...(Object.keys(rateLimit).length > 0 ? { rate_limit: rateLimit } : {}),
      };
      await onSubmit(payload);
    },
  });

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
        <h3 className="text-sm font-semibold">{t('models.form.basicInfo')}</h3>

        <div className="grid grid-cols-2 gap-4">
          <form.Field name="name">
            {(field) => (
              <Field label={t('models.form.nameLabel')}>
                <Input
                  required
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder={t('models.form.namePlaceholder')}
                />
              </Field>
            )}
          </form.Field>

          <form.Field name="model">
            {(field) => (
              <Field
                label={t('models.form.modelLabel')}
                hint={t('models.form.modelHint')}
              >
                <Input
                  required
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder={t('models.form.modelPlaceholder')}
                />
              </Field>
            )}
          </form.Field>
        </div>
      </section>

      {/* Provider Config */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold">
          {t('models.form.providerConfig')}
        </h3>

        <div className="grid grid-cols-2 gap-4">
          <form.Field name="api_key">
            {(field) => (
              <Field label={t('models.form.apiKeyLabel')}>
                <Input
                  type="password"
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder="sk-…"
                  autoComplete="off"
                />
              </Field>
            )}
          </form.Field>

          <form.Field name="api_base">
            {(field) => (
              <Field
                label={t('models.form.apiBase')}
                hint={t('models.form.apiBaseHint')}
              >
                <Input
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder={t('models.form.apiBasePlaceholder')}
                />
              </Field>
            )}
          </form.Field>
        </div>
      </section>

      {/* Advanced */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold text-muted-foreground">
          {t('models.form.advanced')}
        </h3>

        <form.Field name="timeout">
          {(field) => (
            <Field label={t('models.form.timeout')}>
              <Input
                type="number"
                min={0}
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                onBlur={field.handleBlur}
                placeholder="e.g. 30000"
              />
            </Field>
          )}
        </form.Field>

        <div className="border-t pt-4">
          <p className="mb-3 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            {t('models.form.rateLimits')}
          </p>
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
          <Button type="button" variant="outline" onClick={onCancel}>
            {t('common.cancel')}
          </Button>
          <form.Subscribe selector={(s) => s.isSubmitting}>
            {(isSubmitting) => (
              <Button type="submit" disabled={isSubmitting || isPending}>
                {isSubmitting || isPending ? t('common.saving') : submitLabel}
              </Button>
            )}
          </form.Subscribe>
        </div>
      </div>
    </form>
  );
}

// ── Field wrapper ─────────────────────────────────────────────────────────────

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
