import { Link } from '@tanstack/react-router';
import { useForm } from '@tanstack/react-form';
import { Plus, RefreshCw } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxTrigger,
} from '@/components/ui/combobox';
import { Input } from '@/components/ui/input';
import {
  InputGroupAddon,
  InputGroupButton,
  InputGroupText,
} from '@/components/ui/input-group';
import { Label } from '@/components/ui/label';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import type { Model } from '@/lib/api/types';
import { useProviders } from '@/lib/queries/providers';

export interface ModelFormProps {
  initial?: Model;
  onSubmit: (data: Model) => void | Promise<void>;
  onCancel: () => void;
  isPending: boolean;
  error?: string;
  submitLabel: string;
  extraActions?: React.ReactNode;
}

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
  const [clientError, setClientError] = useState<string>();
  const providersQuery = useProviders();
  const providerOptions = (providersQuery.data?.list ?? []).map(
    ({ key, value }) => ({
      value: key.replace('/providers/', ''),
      label: key.replace('/providers/', ''),
      name: value.name,
      type: value.type,
    }),
  );

  const form = useForm({
    defaultValues: {
      name: initial?.name ?? '',
      provider_id: initial?.provider_id ?? '',
      model: initial?.model ?? '',
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
      const trimmedProviderId = value.provider_id.trim();
      if (!trimmedProviderId) {
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
        provider_id: trimmedProviderId,
        model: trimmedModel,
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
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold">{t('models.form.basicInfo')}</h3>

        <div className="grid gap-4 md:grid-cols-3">
          <form.Field name="name">
            {(field) => (
              <Field
                label={t('models.form.nameLabel')}
                className="md:col-span-3"
              >
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

          <form.Field name="provider_id">
            {(field) => {
              const selectedProvider = providerOptions.find(
                (provider) => provider.value === field.state.value.trim(),
              );

              let providerHint = t('models.form.providerSearchHint');
              if (providersQuery.isLoading) {
                providerHint = t('models.form.providerLoading');
              } else if (providersQuery.isError) {
                providerHint = t('models.form.providerLoadError');
              } else if (providerOptions.length === 0) {
                providerHint = t('models.form.providerEmpty');
              } else if (selectedProvider) {
                providerHint = t('models.form.providerSelectedHint', {
                  name: selectedProvider.name,
                  type: selectedProvider.type,
                });
              }

              return (
                <Field
                  label={t('models.form.providerLabel')}
                  hint={providerHint}
                  className="md:col-span-3"
                >
                  <Combobox
                    items={providerOptions}
                    itemToStringLabel={(provider) => provider.label}
                    itemToStringValue={(provider) => provider.value}
                    value={selectedProvider ?? null}
                    inputValue={field.state.value}
                    onValueChange={(provider) => {
                      setClientError(undefined);
                      field.handleChange(provider?.value ?? '');
                    }}
                    onInputValueChange={(inputValue) => {
                      setClientError(undefined);
                      field.handleChange(inputValue);
                    }}
                    autoHighlight
                  >
                    <ComboboxInput
                      showTrigger={false}
                      placeholder={t('models.form.providerPlaceholder')}
                      onBlur={field.handleBlur}
                    >
                      <InputGroupAddon
                        align="inline-end"
                        className="gap-1 pr-1"
                      >
                        {selectedProvider && (
                          <InputGroupText className="max-w-48 truncate pr-1 text-xs">
                            {selectedProvider.name} · {selectedProvider.type}
                          </InputGroupText>
                        )}

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <InputGroupButton
                              asChild
                              size="icon-sm"
                              variant="ghost"
                              aria-label={t(
                                'models.form.toggleProviderOptions',
                              )}
                            >
                              <ComboboxTrigger />
                            </InputGroupButton>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{t('models.form.toggleProviderOptions')}</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <InputGroupButton
                              size="icon-sm"
                              variant="ghost"
                              onClick={() => {
                                void providersQuery.refetch();
                              }}
                              disabled={providersQuery.isFetching}
                              aria-label={t('models.form.refreshProviders')}
                            >
                              <RefreshCw
                                className={
                                  providersQuery.isFetching
                                    ? 'animate-spin'
                                    : undefined
                                }
                              />
                            </InputGroupButton>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{t('models.form.refreshProviders')}</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <InputGroupButton
                              asChild
                              size="icon-sm"
                              variant="ghost"
                            >
                              <Link
                                to="/providers/create"
                                target="_blank"
                                rel="noreferrer"
                              >
                                <Plus />
                              </Link>
                            </InputGroupButton>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{t('providers.addProvider')}</p>
                          </TooltipContent>
                        </Tooltip>
                      </InputGroupAddon>
                    </ComboboxInput>

                    <ComboboxContent>
                      {providersQuery.isLoading ? (
                        <div className="px-3 py-2 text-sm text-muted-foreground">
                          {t('models.form.providerLoading')}
                        </div>
                      ) : providersQuery.isError ? (
                        <div className="px-3 py-2 text-sm text-muted-foreground">
                          {t('models.form.providerLoadError')}
                        </div>
                      ) : providerOptions.length === 0 ? (
                        <div className="px-3 py-2 text-sm text-muted-foreground">
                          {t('models.form.providerEmpty')}
                        </div>
                      ) : (
                        <>
                          <ComboboxEmpty>
                            {t('models.form.providerNoMatch')}
                          </ComboboxEmpty>
                          <ComboboxList>
                            {(provider) => (
                              <ComboboxItem
                                key={provider.value}
                                value={provider}
                                className="flex-col items-start gap-0.5"
                              >
                                <span className="font-mono text-xs text-foreground">
                                  {provider.value}
                                </span>
                                <span className="text-muted-foreground">
                                  {provider.name} · {provider.type}
                                </span>
                              </ComboboxItem>
                            )}
                          </ComboboxList>
                        </>
                      )}
                    </ComboboxContent>
                  </Combobox>
                </Field>
              );
            }}
          </form.Field>

          <form.Field name="model">
            {(field) => (
              <Field
                label={t('models.form.modelLabel')}
                className="md:col-span-3"
              >
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

function Field({
  label,
  hint,
  className,
  children,
}: {
  label: string;
  hint?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={className ? `space-y-1.5 ${className}` : 'space-y-1.5'}>
      <Label className="text-xs font-medium">{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
