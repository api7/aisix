import { useForm } from '@tanstack/react-form';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
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

type ProviderId = 'anthropic' | 'bedrock' | 'deepseek' | 'gemini' | 'openai';
type ProviderSelection = ProviderId | '';

type ProviderConfigValues = Record<string, string>;

interface ProviderConfigFieldSchema {
  type: 'string';
  titleKey: string;
  descriptionKey?: string;
  placeholder?: string;
  placeholderKey?: string;
  inputType?: React.HTMLInputTypeAttribute;
}

interface ProviderConfigSchema {
  required: string[];
  properties: Record<string, ProviderConfigFieldSchema>;
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

const PROVIDER_OPTIONS: Array<{ value: ProviderId; labelKey: string }> = [
  { value: 'openai', labelKey: 'models.form.providers.openai' },
  { value: 'anthropic', labelKey: 'models.form.providers.anthropic' },
  { value: 'gemini', labelKey: 'models.form.providers.gemini' },
  { value: 'deepseek', labelKey: 'models.form.providers.deepseek' },
  { value: 'bedrock', labelKey: 'models.form.providers.bedrock' },
];

const OPENAI_COMPATIBLE_CONFIG_SCHEMA: ProviderConfigSchema = {
  required: ['api_key'],
  properties: {
    api_key: {
      type: 'string',
      titleKey: 'models.form.apiKeyLabel',
      placeholder: 'sk-…',
      inputType: 'password',
    },
    api_base: {
      type: 'string',
      titleKey: 'models.form.apiBase',
      descriptionKey: 'models.form.apiBaseHint',
      placeholderKey: 'models.form.apiBasePlaceholder',
      inputType: 'url',
    },
  },
};

const BEDROCK_CONFIG_SCHEMA: ProviderConfigSchema = {
  required: ['region', 'access_key_id', 'secret_access_key'],
  properties: {
    region: {
      type: 'string',
      titleKey: 'models.form.regionLabel',
      descriptionKey: 'models.form.regionHint',
      placeholder: 'us-east-1',
    },
    access_key_id: {
      type: 'string',
      titleKey: 'models.form.accessKeyIdLabel',
      placeholder: 'AKIA...',
    },
    secret_access_key: {
      type: 'string',
      titleKey: 'models.form.secretAccessKeyLabel',
      inputType: 'password',
      placeholderKey: 'models.form.secretAccessKeyPlaceholder',
    },
    session_token: {
      type: 'string',
      titleKey: 'models.form.sessionTokenLabel',
      descriptionKey: 'models.form.sessionTokenHint',
      inputType: 'password',
      placeholderKey: 'models.form.sessionTokenPlaceholder',
    },
    endpoint: {
      type: 'string',
      titleKey: 'models.form.endpointLabel',
      descriptionKey: 'models.form.endpointHint',
      inputType: 'url',
      placeholder: 'https://bedrock-runtime.us-east-1.amazonaws.com',
    },
  },
};

const PROVIDER_CONFIG_SCHEMAS: Record<ProviderId, ProviderConfigSchema> = {
  anthropic: OPENAI_COMPATIBLE_CONFIG_SCHEMA,
  bedrock: BEDROCK_CONFIG_SCHEMA,
  deepseek: OPENAI_COMPATIBLE_CONFIG_SCHEMA,
  gemini: OPENAI_COMPATIBLE_CONFIG_SCHEMA,
  openai: OPENAI_COMPATIBLE_CONFIG_SCHEMA,
};

function isProviderId(value: string): value is ProviderId {
  return PROVIDER_OPTIONS.some((option) => option.value === value);
}

function splitModelIdentifier(model: string | undefined): {
  provider: ProviderSelection;
  providerModel: string;
} {
  if (!model) {
    return { provider: '', providerModel: '' };
  }

  const separatorIndex = model.indexOf('/');
  if (separatorIndex === -1) {
    return { provider: '', providerModel: model };
  }

  const provider = model.slice(0, separatorIndex).toLowerCase();
  const providerModel = model.slice(separatorIndex + 1);
  if (!isProviderId(provider) || !providerModel) {
    return { provider: '', providerModel: model };
  }

  return { provider, providerModel };
}

function normalizeProviderConfigValues(
  provider: ProviderId,
  source: Model['provider_config'] | ProviderConfigValues | undefined,
): ProviderConfigValues {
  const schema = PROVIDER_CONFIG_SCHEMAS[provider];
  const objectSource =
    source && typeof source === 'object'
      ? (source as Record<string, unknown>)
      : undefined;

  return Object.fromEntries(
    Object.keys(schema.properties).map((fieldName) => {
      const rawValue = objectSource?.[fieldName];
      return [fieldName, typeof rawValue === 'string' ? rawValue : ''];
    }),
  );
}

function serializeProviderConfig(
  provider: ProviderId,
  values: ProviderConfigValues,
): Model['provider_config'] {
  const schema = PROVIDER_CONFIG_SCHEMAS[provider];

  return Object.fromEntries(
    Object.keys(schema.properties)
      .map((fieldName) => [fieldName, values[fieldName]?.trim() ?? ''])
      .filter(([, value]) => value.length > 0),
  );
}

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
  const initialModel = splitModelIdentifier(initial?.model);
  const initialProviderConfigValues = initialModel.provider
    ? normalizeProviderConfigValues(
        initialModel.provider,
        initial?.provider_config,
      )
    : {};
  const [provider, setProvider] = useState<ProviderSelection>(
    initialModel.provider,
  );
  const [providerConfigDrafts, setProviderConfigDrafts] = useState<
    Partial<Record<ProviderId, ProviderConfigValues>>
  >(() =>
    initialModel.provider
      ? { [initialModel.provider]: initialProviderConfigValues }
      : {},
  );
  const [providerConfigValues, setProviderConfigValues] =
    useState<ProviderConfigValues>(initialProviderConfigValues);
  const [clientError, setClientError] = useState<string>();

  const form = useForm({
    defaultValues: {
      name: initial?.name ?? '',
      model: initialModel.providerModel,
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
      if (!provider) {
        setClientError(t('models.form.providerRequired'));
        return;
      }

      const trimmedModel = value.model.trim();
      if (!trimmedModel) {
        setClientError(t('models.form.modelRequired'));
        return;
      }

      setClientError(undefined);

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
        model: `${provider}/${trimmedModel}`,
        provider_config: serializeProviderConfig(
          provider,
          providerConfigValues,
        ),
        ...(timeout != null ? { timeout } : {}),
        ...(Object.keys(rateLimit).length > 0 ? { rate_limit: rateLimit } : {}),
      };
      await onSubmit(payload);
    },
  });

  const providerConfigSchema = provider
    ? PROVIDER_CONFIG_SCHEMAS[provider]
    : undefined;

  function handleProviderChange(nextProvider: string) {
    if (!isProviderId(nextProvider)) {
      return;
    }

    const nextDrafts = provider
      ? {
          ...providerConfigDrafts,
          [provider]: { ...providerConfigValues },
        }
      : providerConfigDrafts;
    const nextProviderDraft = nextDrafts[nextProvider];

    setProviderConfigDrafts(nextDrafts);
    setProvider(nextProvider);
    setProviderConfigValues(
      normalizeProviderConfigValues(nextProvider, nextProviderDraft),
    );
    setClientError(undefined);
  }

  function handleProviderConfigFieldChange(
    fieldName: string,
    nextValue: string,
  ) {
    setProviderConfigValues((current) => {
      const nextValues = {
        ...current,
        [fieldName]: nextValue,
      };

      if (provider) {
        setProviderConfigDrafts((currentDrafts) => ({
          ...currentDrafts,
          [provider]: nextValues,
        }));
      }

      return nextValues;
    });
  }

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

        <div className="grid gap-4 md:grid-cols-3">
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

          <Field label={t('models.form.providerLabel')}>
            <Select
              value={provider || undefined}
              onValueChange={handleProviderChange}
            >
              <SelectTrigger className="w-full">
                <SelectValue
                  placeholder={t('models.form.providerPlaceholder')}
                />
              </SelectTrigger>
              <SelectContent>
                {PROVIDER_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {t(option.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <form.Field name="model">
            {(field) => (
              <Field label={t('models.form.modelLabel')}>
                <Input
                  required
                  value={field.state.value}
                  onChange={(e) => {
                    setClientError(undefined);
                    field.handleChange(e.target.value);
                  }}
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

        {providerConfigSchema ? (
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(providerConfigSchema.properties).map(
              ([fieldName, fieldSchema]) => (
                <Field
                  key={fieldName}
                  label={t(fieldSchema.titleKey)}
                  hint={
                    fieldSchema.descriptionKey
                      ? t(fieldSchema.descriptionKey)
                      : undefined
                  }
                >
                  <Input
                    required={providerConfigSchema.required.includes(fieldName)}
                    type={fieldSchema.inputType ?? 'text'}
                    value={providerConfigValues[fieldName] ?? ''}
                    onChange={(e) =>
                      handleProviderConfigFieldChange(fieldName, e.target.value)
                    }
                    placeholder={
                      fieldSchema.placeholderKey
                        ? t(fieldSchema.placeholderKey)
                        : fieldSchema.placeholder
                    }
                    autoComplete="off"
                  />
                </Field>
              ),
            )}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {t('models.form.providerConfigSelectHint')}
          </p>
        )}
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
          <div className="grid gap-3 md:grid-cols-3">
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

      {(clientError ?? error) && (
        <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {clientError ?? error}
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
