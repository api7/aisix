import { useEffect, useState } from "react";
import { api, AdminApiError } from "../api";
import type { Connection } from "../storage";
import type { Model, ResourceEntry } from "../types";
import { ErrorBanner } from "./ErrorBanner";

interface ModelsViewProps {
  connection: Connection;
}

interface FormState {
  // Either creating (no id) or editing (id from the entry).
  editingId: string | null;
  name: string;
  model: string;
  apiKey: string;
  apiBase: string;
}

const EMPTY_FORM: FormState = {
  editingId: null,
  name: "",
  model: "openai/gpt-4o",
  apiKey: "",
  apiBase: "",
};

export function ModelsView({ connection }: ModelsViewProps) {
  const [entries, setEntries] = useState<ResourceEntry<Model>[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const list = await api.listModels(connection);
      setEntries(list);
    } catch (e) {
      setError(toMessage(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // refresh whenever the connection changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connection.endpoint, connection.adminKey]);

  function startCreate() {
    setForm(EMPTY_FORM);
  }

  function startEdit(entry: ResourceEntry<Model>) {
    setForm({
      editingId: entry.id,
      name: entry.value.name,
      model: entry.value.model,
      apiKey: entry.value.provider_config.api_key,
      apiBase: entry.value.provider_config.api_base ?? "",
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const payload: Model = {
      name: form.name.trim(),
      model: form.model.trim(),
      provider_config: {
        api_key: form.apiKey,
        api_base: form.apiBase.trim() === "" ? undefined : form.apiBase.trim(),
      },
    };
    try {
      if (form.editingId) {
        await api.updateModel(connection, form.editingId, payload);
      } else {
        await api.createModel(connection, payload);
      }
      setForm(EMPTY_FORM);
      await refresh();
    } catch (e) {
      setError(toMessage(e));
    }
  }

  async function handleDelete(id: string) {
    if (!confirm(`Delete model ${id}? This is permanent.`)) return;
    setError(null);
    try {
      await api.deleteModel(connection, id);
      if (form.editingId === id) setForm(EMPTY_FORM);
      await refresh();
    } catch (e) {
      setError(toMessage(e));
    }
  }

  return (
    <section className="space-y-6">
      <header className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold tracking-tight">Models</h2>
        <button
          type="button"
          onClick={startCreate}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          New
        </button>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <form onSubmit={handleSubmit} className="space-y-3 rounded-md border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="grid grid-cols-2 gap-3">
          <label className="block">
            <span className="text-sm font-medium">Display name</span>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              required
              className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium">Provider/model</span>
            <input
              type="text"
              value={form.model}
              onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))}
              required
              placeholder="openai/gpt-4o"
              className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium">API key</span>
            <input
              type="password"
              value={form.apiKey}
              onChange={(e) => setForm((f) => ({ ...f, apiKey: e.target.value }))}
              required
              className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium">API base (optional)</span>
            <input
              type="text"
              value={form.apiBase}
              onChange={(e) => setForm((f) => ({ ...f, apiBase: e.target.value }))}
              className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            />
          </label>
        </div>
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
        <p className="text-sm text-zinc-500">No models yet.</p>
      ) : (
        <ul className="divide-y divide-zinc-200 rounded-md border border-zinc-200 bg-white dark:divide-zinc-800 dark:border-zinc-800 dark:bg-zinc-900">
          {entries.map((e) => (
            <li
              key={e.id}
              className="flex items-center justify-between gap-4 px-4 py-3"
            >
              <div className="min-w-0">
                <div className="font-medium">{e.value.name}</div>
                <div className="truncate text-xs text-zinc-500">
                  {e.value.model} · rev {e.revision}
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
