import { useForm } from '@tanstack/react-form';
import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { X } from 'lucide-react';
import { useState } from 'react';

import { PageHeader } from '@/components/layout/page-header';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { ApiKey } from '@/lib/api/types';
import { useCreateApiKey } from '@/lib/queries/apikeys';
import { useModels } from '@/lib/queries/models';

export const Route = createFileRoute('/_layout/apikeys/create')({
  component: ApiKeyCreatePage,
});

function ApiKeyCreatePage() {
  const navigate = useNavigate();
  const createApiKey = useCreateApiKey();

  async function handleSubmit(data: ApiKey) {
    await createApiKey.mutateAsync(data);
    navigate({ to: '/apikeys' });
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">API Key</h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate({ to: '/apikeys' })}
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl space-y-6">
          <div>
            <h2 className="text-base font-semibold">Create API key resource</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Required: key, allowed_models. Optional: rate_limit.
            </p>
          </div>

          <ApiKeyForm
            onSubmit={handleSubmit}
            isPending={createApiKey.isPending}
            error={createApiKey.error?.message}
            onCancel={() => navigate({ to: '/apikeys' })}
            submitLabel="Create API Key"
          />
        </div>
      </div>
    </div>
  );
}

// ── Shared form component ─────────────────────────────────────────────────────
interface ApiKeyFormProps {
  initial?: ApiKey;
  onSubmit: (data: ApiKey) => void | Promise<void>;
  onCancel: () => void;
  isPending: boolean;
  error?: string;
  submitLabel: string;
  extraActions?: React.ReactNode;
}

const RATE_LIMIT_FIELDS = [
  { name: 'tpm' as const, label: 'TPM', hint: 'Tokens / minute' },
  { name: 'tpd' as const, label: 'TPD', hint: 'Tokens / day' },
  { name: 'rpm' as const, label: 'RPM', hint: 'Requests / minute' },
  { name: 'rpd' as const, label: 'RPD', hint: 'Requests / day' },
  { name: 'concurrency' as const, label: 'Concurrency', hint: undefined },
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
  const { data: modelsData } = useModels();
  const modelOptions = modelsData?.list ?? [];

  // allowedModels managed separately — it's a dynamic multi-value list
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
        <h3 className="text-sm font-semibold">Basic Information</h3>

        <form.Field name="key">
          {(field) => (
            <Field label="API Key *">
              <Input
                required
                value={field.state.value}
                onChange={(e) => field.handleChange(e.target.value)}
                onBlur={field.handleBlur}
                placeholder="sk-…"
                autoComplete="off"
              />
              <p className="text-xs text-muted-foreground">
                This value will be matched against incoming Bearer tokens.
              </p>
            </Field>
          )}
        </form.Field>
      </section>

      {/* Allowed Models */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold">Allowed Models</h3>
        <p className="text-xs text-muted-foreground">
          Leave empty to allow all models.
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
            placeholder="Type or select a model key…"
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
            Add
          </Button>
        </div>

        {allowedModels.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {allowedModels.map((m) => (
              <Badge
                key={m}
                variant="secondary"
                className="cursor-pointer font-mono text-xs"
                onClick={() => removeModel(m)}
                title="Click to remove"
              >
                {m} ×
              </Badge>
            ))}
          </div>
        )}
      </section>

      {/* Rate limits */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Rate Limits (Optional)
        </h3>

        <div className="grid grid-cols-3 gap-3">
          {RATE_LIMIT_FIELDS.map(({ name, label, hint }) => (
            <form.Field key={name} name={name}>
              {(field) => (
                <Field label={label} hint={hint}>
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
            Cancel
          </Button>
          <form.Subscribe selector={(s) => s.isSubmitting}>
            {(isSubmitting) => (
              <Button
                type="submit"
                size="lg"
                disabled={isSubmitting || isPending}
              >
                {isSubmitting || isPending ? 'Saving…' : submitLabel}
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
