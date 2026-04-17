// aisix — Admin UI entrypoint. Scaffold for PR #1.
// Real layout, routing, theme, i18n arrive in PR #11 (see plan §4.11).

export default function App() {
  return (
    <main className="min-h-dvh flex items-center justify-center bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <section className="max-w-xl px-8 py-12">
        <h1 className="text-5xl font-semibold tracking-tight">aisix</h1>
        <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
          AI Gateway scaffold. Admin console is arriving in PR #11 — models, API
          keys, playground, observability, and the full bento dashboard.
        </p>
        <pre className="mt-8 rounded-md bg-zinc-900 text-zinc-100 px-4 py-3 text-xs overflow-x-auto">
{`GET  /aisix/admin/health
POST /aisix/admin/models
POST /playground/chat/completions`}
        </pre>
      </section>
    </main>
  );
}
