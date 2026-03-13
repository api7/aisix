import { createContext, useContext } from 'react';

interface SidebarContextValue {
  isOpen: boolean;
  toggle: () => void;
  open: () => void;
  close: () => void;
}

export const SidebarContext = createContext<SidebarContextValue>({
  isOpen: false,
  toggle: () => {},
  open: () => {},
  close: () => {},
});

export function useSidebar(): SidebarContextValue {
  return useContext(SidebarContext);
}
