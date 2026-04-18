import { useState } from "react";
import {
  type Connection,
  DEFAULT_CONNECTION,
  saveConnection,
} from "../storage";

interface ConnectionFormProps {
  initial: Connection;
  onSaved: (c: Connection) => void;
}

// Connection settings form. Stores the admin endpoint + key in
// localStorage via `saveConnection` so the user doesn't have to
// re-enter on reload. Validation is intentionally light — the API
// itself will reject bad endpoints and the {error_msg} envelope
// surfaces through ErrorBanner.
export function ConnectionForm({ initial, onSaved }: ConnectionFormProps) {
  const [endpoint, setEndpoint] = useState(initial.endpoint);
  const [adminKey, setAdminKey] = useState(initial.adminKey);
  const [validationError, setValidationError] = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmedEndpoint = endpoint.trim();
    const trimmedKey = adminKey.trim();
    if (!trimmedEndpoint) {
      setValidationError("Endpoint is required.");
      return;
    }
    if (!trimmedKey) {
      setValidationError("Admin key is required.");
      return;
    }
    setValidationError(null);
    const next: Connection = {
      endpoint: trimmedEndpoint,
      adminKey: trimmedKey,
    };
    saveConnection(next);
    onSaved(next);
  }

  function handleReset() {
    setEndpoint(DEFAULT_CONNECTION.endpoint);
    setAdminKey("");
    setValidationError(null);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-xl">
      <h2 className="text-2xl font-semibold tracking-tight">Connection</h2>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">
        Where to reach the aisix admin listener and which admin key to use.
        Stored in your browser only.
      </p>

      <label className="block">
        <span className="text-sm font-medium">Admin endpoint</span>
        <input
          type="text"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          placeholder="http://127.0.0.1:3001"
          className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
      </label>

      <label className="block">
        <span className="text-sm font-medium">Admin key</span>
        <input
          type="password"
          value={adminKey}
          onChange={(e) => setAdminKey(e.target.value)}
          placeholder="Bearer token from cfg.admin.admin_keys"
          className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
      </label>

      {validationError && (
        <p role="alert" className="text-sm text-red-700 dark:text-red-300">
          {validationError}
        </p>
      )}

      <div className="flex gap-3">
        <button
          type="submit"
          className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-50 hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
        >
          Save
        </button>
        <button
          type="button"
          onClick={handleReset}
          className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          Reset
        </button>
      </div>
    </form>
  );
}
