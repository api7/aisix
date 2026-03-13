import { useForm } from '@tanstack/react-form';
import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { X } from 'lucide-react';

import { PageHeader } from '@/components/layout/page-header';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { Model } from '@/lib/api/types';
import { useCreateModel } from '@/lib/queries/models';

export const Route = createFileRoute('/_layout/models/create')({
  component: ModelCreatePage,
});

function ModelCreatePage() {
  const navigate = useNavigate();
  const createModel = useCreateModel();

  async function handleSubmit(data: Model) {
    await createModel.mutateAsync(data);
    navigate({ to: '/models' });
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">Model</h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate({ to: '/models' })}
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl space-y-6">
          <div>
            <h2 className="text-base font-semibold">Create model resource</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Required: name, model, provider config API key. Optional: timeout,
              rate limits.
            </p>
          </div>

          <ModelForm
            onSubmit={handleSubmit}
            isPending={createModel.isPending}
            error={createModel.error?.message}
            onCancel={() => navigate({ to: '/models' })}
            submitLabel="Create Model"
          />
        </div>
      </div>
    </div>
  );
}

// ── Shared form component ─────────────────────────────────────────────────────
interface ModelFormProps {
  initial?: Model;
  onSubmit: (data: Model) => void | Promise<void>;
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

export function ModelForm({
  initial,
  onSubmit,
  onCancel,
  isPending,
  error,
  submitLabel,
  extraActions,
}: ModelFormProps) {
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
      const rateLimit = {
        ...(value.tpm ? { tpm: Number(value.tpm) } : {}),
        ...(value.tpd ? { tpd: Number(value.tpd) } : {}),
        ...(value.rpm ? { rpm: Number(value.rpm) } : {}),
        ...(value.rpd ? { rpd: Number(value.rpd) } : {}),
        ...(value.concurrency
          ? { concurrency: Number(value.concurrency) }
          : {}),
      };
      const payload: Model = {
        name: value.name.trim(),
        model: value.model.trim(),
        provider_config: {
          ...(value.api_key ? { api_key: value.api_key.trim() } : {}),
          ...(value.api_base ? { api_base: value.api_base.trim() } : {}),
        },
        ...(value.timeout ? { timeout: Number(value.timeout) } : {}),
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
        <h3 className="text-sm font-semibold">Basic Information</h3>

        <div className="grid grid-cols-2 gap-4">
          <form.Field name="name">
            {(field) => (
              <Field label="Name *">
                <Input
                  required
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder="e.g. @my-llm/chat"
                />
              </Field>
            )}
          </form.Field>

          <form.Field name="model">
            {(field) => (
              <Field label="Model *" hint="Format: provider/model-name">
                <Input
                  required
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder="e.g. deepseek/deepseek-chat"
                />
              </Field>
            )}
          </form.Field>
        </div>
      </section>

      {/* Provider Config */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold">Provider Config</h3>

        <div className="grid grid-cols-2 gap-4">
          <form.Field name="api_key">
            {(field) => (
              <Field label="API Key">
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
                label="API Base"
                hint="Leave blank to use provider default"
              >
                <Input
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  onBlur={field.handleBlur}
                  placeholder="https://api.example.com/v1"
                />
              </Field>
            )}
          </form.Field>
        </div>
      </section>

      {/* Advanced */}
      <section className="space-y-4 rounded-xl border bg-card p-5">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Advanced (Optional)
        </h3>

        <form.Field name="timeout">
          {(field) => (
            <Field label="Timeout (ms)">
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
            Rate Limits
          </p>
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
            Cancel
          </Button>
          <form.Subscribe selector={(s) => s.isSubmitting}>
            {(isSubmitting) => (
              <Button type="submit" disabled={isSubmitting || isPending}>
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
