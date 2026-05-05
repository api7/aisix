import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { useAdminKey } from '@/hooks/use-admin-key';
import { ApiClientError, providersApi } from '@/lib/api/client';
import type { Provider } from '@/lib/api/types';

export const providerKeys = {
  all: ['providers'] as const,
  list: () => [...providerKeys.all, 'list'] as const,
  detail: (id: string) => [...providerKeys.all, 'detail', id] as const,
};

export function useProviders() {
  const { key, openModal } = useAdminKey();
  return useQuery({
    queryKey: providerKeys.list(),
    queryFn: () => providersApi.list(key!),
    enabled: !!key,
    retry: (count, err) => {
      if (err instanceof ApiClientError && err.status === 401) {
        openModal();
        return false;
      }
      return count < 2;
    },
  });
}

export function useProvider(id: string) {
  const { key, openModal } = useAdminKey();
  return useQuery({
    queryKey: providerKeys.detail(id),
    queryFn: () => providersApi.get(key!, id),
    enabled: !!key && !!id,
    retry: (count, err) => {
      if (err instanceof ApiClientError && err.status === 401) {
        openModal();
        return false;
      }
      return count < 2;
    },
  });
}

export function useCreateProvider() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (data: Provider) => providersApi.create(key!, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: providerKeys.list() }),
  });
}

export function useUpdateProvider(id: string) {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (data: Provider) => providersApi.update(key!, id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: providerKeys.list() });
      qc.invalidateQueries({ queryKey: providerKeys.detail(id) });
    },
  });
}

export function useDeleteProvider() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (id: string) => providersApi.delete(key!, id),
    onSuccess: () => qc.invalidateQueries({ queryKey: providerKeys.list() }),
  });
}
