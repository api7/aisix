import { useForm } from '@tanstack/react-form';
import { Eye, EyeOff } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { PROVIDER_TYPE_VARIANTS } from '@/lib/api/types';
import type { Provider, ProviderType } from '@/lib/api/types';

export interface ProviderFormProps {
  initial?: Provider;
  onSubmit: (data: Provider) => void | Promise<void>;
  onCancel: () => void;
  isPending: boolean;
  error?: string;
  submitLabel: string;
  extraActions?: React.ReactNode;
}

const PROVIDER_TYPES = Array.from(PROVIDER_TYPE_VARIANTS);

function trimOptional(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

export function ProviderForm({
  initial,
  onSubmit,
  onCancel,
  isPending,
  error,
  submitLabel,
  extraActions,
}: ProviderFormProps) {
  const { t } = useTranslation();
  const [clientError, setClientError] = useState<string>();

  const form = useForm({
    defaultValues: {
      name: initial?.name ?? '',
      type: initial?.type ?? ('openai' as ProviderType),
      api_key:
        initial?.type !== 'bedrock' ? (initial?.config.api_key ?? '') : '',
      api_base:
        initial?.type !== 'bedrock' ? (initial?.config.api_base ?? '') : '',
      api_version:
        initial?.type === 'azure' ? (initial.config.api_version ?? '') : '',
      region: initial?.type === 'bedrock' ? initial.config.region : '',
      access_key_id:
        initial?.type === 'bedrock' ? initial.config.access_key_id : '',
      secret_access_key:
        initial?.type === 'bedrock' ? initial.config.secret_access_key : '',
      session_token:
        initial?.type === 'bedrock' ? (initial.config.session_token ?? '') : '',
      endpoint:
        initial?.type === 'bedrock' ? (initial.config.endpoint ?? '') : '',
    },
    onSubmit: async ({ value }) => {
      const name = value.name.trim();
      if (!name) {
        setClientError(t('providers.form.nameRequired'));
        return;
      }

      if (value.type === 'bedrock') {
        const region = value.region.trim();
        const accessKeyId = value.access_key_id.trim();
        const secretAccessKey = value.secret_access_key.trim();

        if (!region) {
          setClientError(t('providers.form.regionRequired'));
          return;
        }

        if (!accessKeyId) {
          setClientError(t('providers.form.accessKeyIdRequired'));
          return;
        }

        if (!secretAccessKey) {
          setClientError(t('providers.form.secretAccessKeyRequired'));
          return;
        }

        setClientError(undefined);
        await onSubmit({
          name,
          type: 'bedrock',
          config: {
            region,
            access_key_id: accessKeyId,
            secret_access_key: secretAccessKey,
            ...(trimOptional(value.session_token)
              ? { session_token: trimOptional(value.session_token) }
              : {}),
            ...(trimOptional(value.endpoint)
              ? { endpoint: trimOptional(value.endpoint) }
              : {}),
          },
        });
        return;
      }

      const apiKey = value.api_key.trim();
      if (!apiKey) {
        setClientError(t('providers.form.apiKeyRequired'));
        return;
      }

      const apiBase = trimOptional(value.api_base);

      if (value.type === 'azure') {
        if (!apiBase) {
          setClientError(t('providers.form.apiBaseRequired'));
          return;
        }

        setClientError(undefined);
        await onSubmit({
          name,
          type: 'azure',
          config: {
            api_key: apiKey,
            api_base: apiBase,
            ...(trimOptional(value.api_version)
              ? { api_version: trimOptional(value.api_version) }
              : {}),
          },
        });
        return;
      }

      setClientError(undefined);
      await onSubmit({
        name,
        type: value.type,
        config: {
          api_key: apiKey,
          ...(apiBase ? { api_base: apiBase } : {}),
        },
      });
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
        <h3 className="text-sm font-semibold">
          {t('providers.form.basicInfo')}
        </h3>

        <div className="grid gap-4 md:grid-cols-2">
          <form.Field name="name">
            {(field) => (
              <Field label={t('providers.form.nameLabel')}>
                <Input
                  required
                  value={field.state.value}
                  onChange={(e) => {
                    setClientError(undefined);
                    field.handleChange(e.target.value);
                  }}
                  onBlur={field.handleBlur}
                  placeholder={t('providers.form.namePlaceholder')}
                />
              </Field>
            )}
          </form.Field>

          <form.Field name="type">
            {(field) => (
              <Field label={t('providers.form.typeLabel')}>
                <Select
                  value={field.state.value}
                  onValueChange={(next) => {
                    setClientError(undefined);
                    field.handleChange(next as ProviderType);
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={t('providers.form.typePlaceholder')}
                    />
                  </SelectTrigger>
                  <SelectContent align="start" position="popper">
                    {PROVIDER_TYPES.map((providerType) => (
                      <SelectItem key={providerType} value={providerType}>
                        {t(`providers.form.types.${providerType}`)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            )}
          </form.Field>
        </div>
      </section>

      <form.Subscribe selector={(state) => state.values.type}>
        {(providerType) => (
          <section className="space-y-4 rounded-xl border bg-card p-5">
            <h3 className="text-sm font-semibold">
              {t('providers.form.providerConfig')}
            </h3>

            {providerType === 'bedrock' ? (
              <div className="grid gap-4 md:grid-cols-2">
                <form.Field name="region">
                  {(field) => (
                    <Field
                      label={t('providers.form.regionLabel')}
                      hint={t('providers.form.regionHint')}
                    >
                      <Input
                        required
                        value={field.state.value}
                        onChange={(e) => {
                          setClientError(undefined);
                          field.handleChange(e.target.value);
                        }}
                        onBlur={field.handleBlur}
                        placeholder="us-east-1"
                      />
                    </Field>
                  )}
                </form.Field>

                <form.Field name="endpoint">
                  {(field) => (
                    <Field
                      label={t('providers.form.endpointLabel')}
                      hint={t('providers.form.endpointHint')}
                    >
                      <Input
                        value={field.state.value}
                        onChange={(e) => field.handleChange(e.target.value)}
                        onBlur={field.handleBlur}
                        placeholder="https://bedrock-runtime.us-east-1.amazonaws.com"
                      />
                    </Field>
                  )}
                </form.Field>

                <form.Field name="access_key_id">
                  {(field) => (
                    <Field label={t('providers.form.accessKeyIdLabel')}>
                      <Input
                        required
                        value={field.state.value}
                        onChange={(e) => {
                          setClientError(undefined);
                          field.handleChange(e.target.value);
                        }}
                        onBlur={field.handleBlur}
                        placeholder="AKIA..."
                        autoComplete="off"
                      />
                    </Field>
                  )}
                </form.Field>

                <form.Field name="secret_access_key">
                  {(field) => (
                    <Field label={t('providers.form.secretAccessKeyLabel')}>
                      <SecretInput
                        required
                        value={field.state.value}
                        onChange={(e) => {
                          setClientError(undefined);
                          field.handleChange(e.target.value);
                        }}
                        onBlur={field.handleBlur}
                        placeholder={t(
                          'providers.form.secretAccessKeyPlaceholder',
                        )}
                        autoComplete="new-password"
                        showLabel={t('providers.form.showSecret')}
                        hideLabel={t('providers.form.hideSecret')}
                      />
                    </Field>
                  )}
                </form.Field>

                <form.Field name="session_token">
                  {(field) => (
                    <Field
                      label={t('providers.form.sessionTokenLabel')}
                      hint={t('providers.form.sessionTokenHint')}
                    >
                      <Input
                        value={field.state.value}
                        onChange={(e) => field.handleChange(e.target.value)}
                        onBlur={field.handleBlur}
                        placeholder={t(
                          'providers.form.sessionTokenPlaceholder',
                        )}
                        autoComplete="off"
                      />
                    </Field>
                  )}
                </form.Field>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                <form.Field name="api_key">
                  {(field) => (
                    <Field label={t('providers.form.apiKeyLabel')}>
                      <SecretInput
                        required
                        value={field.state.value}
                        onChange={(e) => {
                          setClientError(undefined);
                          field.handleChange(e.target.value);
                        }}
                        onBlur={field.handleBlur}
                        placeholder="sk-..."
                        autoComplete="new-password"
                        showLabel={t('providers.form.showSecret')}
                        hideLabel={t('providers.form.hideSecret')}
                      />
                    </Field>
                  )}
                </form.Field>

                <form.Field name="api_base">
                  {(field) => (
                    <Field
                      label={
                        providerType === 'azure'
                          ? `${t('providers.form.apiBase')} *`
                          : t('providers.form.apiBase')
                      }
                      hint={t(
                        providerType === 'azure'
                          ? 'providers.form.azureApiBaseHint'
                          : 'providers.form.apiBaseHint',
                      )}
                    >
                      <Input
                        value={field.state.value}
                        onChange={(e) => {
                          setClientError(undefined);
                          field.handleChange(e.target.value);
                        }}
                        onBlur={field.handleBlur}
                        placeholder={t(
                          providerType === 'azure'
                            ? 'providers.form.azureApiBasePlaceholder'
                            : 'providers.form.apiBasePlaceholder',
                        )}
                      />
                    </Field>
                  )}
                </form.Field>

                {providerType === 'azure' && (
                  <form.Field name="api_version">
                    {(field) => (
                      <Field
                        label={t('providers.form.apiVersionLabel')}
                        hint={t('providers.form.apiVersionHint')}
                      >
                        <Input
                          value={field.state.value}
                          onChange={(e) => {
                            setClientError(undefined);
                            field.handleChange(e.target.value);
                          }}
                          onBlur={field.handleBlur}
                          placeholder={t(
                            'providers.form.apiVersionPlaceholder',
                          )}
                        />
                      </Field>
                    )}
                  </form.Field>
                )}
              </div>
            )}
          </section>
        )}
      </form.Subscribe>

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
          <form.Subscribe selector={(state) => state.isSubmitting}>
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

type SecretInputProps = Omit<
  React.ComponentProps<typeof InputGroupInput>,
  'type'
> & {
  showLabel: string;
  hideLabel: string;
};

function SecretInput({
  showLabel,
  hideLabel,
  disabled,
  ...props
}: SecretInputProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <InputGroup>
      <InputGroupInput
        {...props}
        disabled={disabled}
        type={isVisible ? 'text' : 'password'}
      />
      <InputGroupAddon align="inline-end" className="pr-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <InputGroupButton
              type="button"
              variant="ghost"
              size="icon-sm"
              disabled={disabled}
              aria-label={isVisible ? hideLabel : showLabel}
              onClick={() => setIsVisible((visible) => !visible)}
            >
              {isVisible ? <EyeOff /> : <Eye />}
            </InputGroupButton>
          </TooltipTrigger>
          <TooltipContent>
            <p>{isVisible ? hideLabel : showLabel}</p>
          </TooltipContent>
        </Tooltip>
      </InputGroupAddon>
    </InputGroup>
  );
}
