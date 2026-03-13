import { createFileRoute } from '@tanstack/react-router';

import { PageHeader } from '@/components/layout/page-header';

export const Route = createFileRoute('/_layout/settings')({
  component: SettingsPage,
});

function SettingsPage() {
  return (
    <div className="flex h-full flex-col">
      <PageHeader>
        <h1 className="flex-1 text-xl font-semibold">Settings</h1>
      </PageHeader>

      <div className="flex-1 overflow-auto bg-muted/20 p-5">
        <div className="mx-auto max-w-3xl rounded-xl border bg-card p-6 text-sm text-muted-foreground">
          Settings content coming soon.
        </div>
      </div>
    </div>
  );
}
