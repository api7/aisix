import { useEffect, useState } from "react";
import { api, AdminApiError } from "../api";
import type { Connection } from "../storage";
import type { ApiKey, ResourceEntry } from "../types";
import { ErrorBanner } from "./ErrorBanner";

interface ApiKeysViewProps {
  connection: Connection;
}

interface FormState {
  editingId: string | null;
  key: string;
  // Comma-separated for the form; split on save.
  allowedModelsCsv: string;
}

const EMPTY_FORM: FormState = {
  editingId: null,
  key: "",
  allowedModelsCsv: "",
};

export function ApiKeysView({ connection }: ApiKeysViewProps) {
  const [entries, setEntries] = useState<ResourceEntry<ApiKey>[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const list = await api.listApiKeys(connection);
      setEntries(list);
    } catch (e) {
      setError(toMessage(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connection.endpoint, connection.adminKey]);

  function startCreate() {
    setForm(EMPTY_FORM);
  }

  function startEdit(entry: ResourceEntry<ApiKey>) {
    setForm({
      editingId: entry.id,
      key: entry.value.key,
      allowedModelsCsv: entry.value.allowed_models.join(", "),
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const allowed = form.allowedModelsCsv
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const payload: ApiKey = {
      key: form.key.trim(),
      allowed_models: allowed,
    };
    try {
      if (form.editingId) {
        await api.updateApiKey(connection, form.editingId, payload);
      } else {
        await api.createApiKey(connection, payload);
      }
      setForm(EMPTY_FORM);
      await refresh();
    } catch (e) {
      setError(toMessage(e));
    }
  }

  async function handleDelete(id: string) {
    if (!confirm(`Delete API key ${id}? This is permanent.`)) return;
    setError(null);
    try {
      await api.deleteApiKey(connection, id);
      if (form.editingId === id) setForm(EMPTY_FORM);
      await refresh();
    } catch (e) {
      setError(toMessage(e));
    }
  }

  return (
    <section className="space-y-6">
      <header className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold tracking-tight">API keys</h2>
        <button
          type="button"
          onClick={startCreate}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          New
        </button>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <form
        onSubmit={handleSubmit}
        className="space-y-3 rounded-md border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
      >
        <label className="block">
          <span className="text-sm font-medium">Key</span>
          <input
            type="text"
            value={form.key}
            onChange={(e) => setForm((f) => ({ ...f, key: e.target.value }))}
            required
            placeholder="sk-..."
            className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
        </label>
        <label className="block">
          <span className="text-sm font-medium">Allowed models (comma-separated)</span>
          <input
            type="text"
            value={form.allowedModelsCsv}
            onChange={(e) =>
              setForm((f) => ({ ...f, allowedModelsCsv: e.target.value }))
            }
            placeholder="my-gpt4, my-claude"
            className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          />
          <span className="mt-1 block text-xs text-zinc-500">
            Empty list denies all models (per spec §3).
          </span>
        </label>
        <div className="flex gap-2">
          <button
            type="submit"
            className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-zinc-50 hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            {form.editingId ? "Update" : "Create"}
          </button>
          {form.editingId && (
            <button
              type="button"
              onClick={() => setForm(EMPTY_FORM)}
              className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              Cancel
            </button>
          )}
        </div>
      </form>

      {loading ? (
        <p className="text-sm text-zinc-500">Loading…</p>
      ) : entries.length === 0 ? (
        <p className="text-sm text-zinc-500">No API keys yet.</p>
      ) : (
        <ul className="divide-y divide-zinc-200 rounded-md border border-zinc-200 bg-white dark:divide-zinc-800 dark:border-zinc-800 dark:bg-zinc-900">
          {entries.map((e) => (
            <li
              key={e.id}
              className="flex items-center justify-between gap-4 px-4 py-3"
            >
              <div className="min-w-0">
                <div className="font-mono text-sm">{e.value.key}</div>
                <div className="truncate text-xs text-zinc-500">
                  {e.value.allowed_models.length === 0
                    ? "no models allowed"
                    : `models: ${e.value.allowed_models.join(", ")}`}
                  {" · rev "}
                  {e.revision}
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => startEdit(e)}
                  className="rounded-md border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                >
                  Edit
                </button>
                <button
                  type="button"
                  onClick={() => handleDelete(e.id)}
                  className="rounded-md border border-red-300 px-2 py-1 text-xs text-red-700 hover:bg-red-50 dark:border-red-700 dark:text-red-300 dark:hover:bg-red-950"
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function toMessage(e: unknown): string {
  if (e instanceof AdminApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Unknown error";
}
