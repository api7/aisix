import { Outlet, createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';

import { AdminKeyModal } from '@/components/layout/admin-key-modal';
import { DashboardSidebar } from '@/components/layout/sidebar';
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet';
import {
  AdminKeyContext,
  getStoredAdminKey,
  setStoredAdminKey,
} from '@/hooks/use-admin-key';
import { SidebarContext } from '@/hooks/use-sidebar';

export const Route = createFileRoute('/_layout')({
  component: DashboardLayout,
});

function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [adminKey, setAdminKeyState] = useState<string | null>(() =>
    getStoredAdminKey(),
  );
  const [keyModalOpen, setKeyModalOpen] = useState(() => !getStoredAdminKey());

  function handleSetKey(k: string) {
    setStoredAdminKey(k);
    setAdminKeyState(k);
  }

  return (
    <AdminKeyContext.Provider
      value={{
        key: adminKey,
        setKey: handleSetKey,
        openModal: () => setKeyModalOpen(true),
        closeModal: () => setKeyModalOpen(false),
        isModalOpen: keyModalOpen,
      }}
    >
      <SidebarContext.Provider
        value={{
          isOpen: sidebarOpen,
          toggle: () => setSidebarOpen((v) => !v),
          open: () => setSidebarOpen(true),
          close: () => setSidebarOpen(false),
        }}
      >
        <div className="flex h-screen overflow-hidden bg-background">
          {/* Desktop sidebar — always visible on md+ */}
          <aside className="hidden w-60 flex-none border-r md:block">
            <DashboardSidebar />
          </aside>

          {/* Mobile sidebar — Sheet drawer triggered by PageHeader toggle */}
          <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
            <SheetContent side="left" className="w-60 p-0">
              <SheetTitle className="sr-only">Navigation</SheetTitle>
              <DashboardSidebar />
            </SheetContent>
          </Sheet>

          {/* Main content — each page owns its header and body */}
          <main className="flex flex-1 flex-col overflow-hidden">
            <Outlet />
          </main>
        </div>

        <AdminKeyModal />
      </SidebarContext.Provider>
    </AdminKeyContext.Provider>
  );
}
