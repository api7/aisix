import { useState } from "react";
import { ApiKeysView } from "./components/ApiKeysView";
import { ConnectionForm } from "./components/ConnectionForm";
import { ModelsView } from "./components/ModelsView";
import { isConfigured, loadConnection } from "./storage";

type Tab = "models" | "apikeys" | "connection";

export default function App() {
  const [connection, setConnection] = useState(loadConnection());
  const [tab, setTab] = useState<Tab>(
    isConfigured(loadConnection()) ? "models" : "connection",
  );

  return (
    <div className="min-h-dvh bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <header className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">aisix admin</h1>
          <nav className="flex gap-2">
            <NavButton label="Models" active={tab === "models"} onClick={() => setTab("models")} />
            <NavButton label="API keys" active={tab === "apikeys"} onClick={() => setTab("apikeys")} />
            <NavButton label="Connection" active={tab === "connection"} onClick={() => setTab("connection")} />
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {tab === "connection" && (
          <ConnectionForm
            initial={connection}
            onSaved={(c) => {
              setConnection(c);
              setTab("models");
            }}
          />
        )}
        {tab === "models" && <ModelsView connection={connection} />}
        {tab === "apikeys" && <ApiKeysView connection={connection} />}
      </main>
    </div>
  );
}

interface NavButtonProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

function NavButton({ label, active, onClick }: NavButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        active
          ? "rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-zinc-50 dark:bg-zinc-100 dark:text-zinc-900"
          : "rounded-md px-3 py-1.5 text-sm font-medium hover:bg-zinc-100 dark:hover:bg-zinc-800"
      }
    >
      {label}
    </button>
  );
}
