import { queryClient } from '@/lib/queries';
import { createContext, useContext } from 'react';

const STORAGE_KEY = 'admin_key';

export function getStoredAdminKey(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(STORAGE_KEY);
}

export function setStoredAdminKey(key: string): void {
  localStorage.setItem(STORAGE_KEY, key);
  queryClient.clear();
}

export function clearStoredAdminKey(): void {
  localStorage.removeItem(STORAGE_KEY);
  queryClient.clear();
}

interface AdminKeyContextValue {
  key: string | null;
  setKey: (k: string) => void;
  openModal: () => void;
  closeModal: () => void;
  isModalOpen: boolean;
}

export const AdminKeyContext = createContext<AdminKeyContextValue>({
  key: null,
  setKey: () => {},
  openModal: () => {},
  closeModal: () => {},
  isModalOpen: false,
});

export function useAdminKey(): AdminKeyContextValue {
  return useContext(AdminKeyContext);
}
