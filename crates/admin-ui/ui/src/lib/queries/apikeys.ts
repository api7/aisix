import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { useAdminKey } from '@/hooks/use-admin-key';
import { ApiClientError, apiKeysApi } from '@/lib/api/client';
import type { ApiKey } from '@/lib/api/types';

export const apiKeyKeys = {
  all: ['apikeys'] as const,
  list: () => [...apiKeyKeys.all, 'list'] as const,
  detail: (id: string) => [...apiKeyKeys.all, 'detail', id] as const,
};

export function useApiKeys() {
  const { key, openModal } = useAdminKey();
  return useQuery({
    queryKey: apiKeyKeys.list(),
    queryFn: () => apiKeysApi.list(key!),
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

export function useApiKey(id: string) {
  const { key, openModal } = useAdminKey();
  return useQuery({
    queryKey: apiKeyKeys.detail(id),
    queryFn: () => apiKeysApi.get(key!, id),
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

export function useCreateApiKey() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (data: ApiKey) => apiKeysApi.create(key!, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: apiKeyKeys.list() }),
  });
}

export function useUpdateApiKey() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: ApiKey }) =>
      apiKeysApi.update(key!, id, data),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: apiKeyKeys.list() });
      qc.invalidateQueries({ queryKey: apiKeyKeys.detail(id) });
    },
  });
}

export function useDeleteApiKey() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (id: string) => apiKeysApi.delete(key!, id),
    onSuccess: () => qc.invalidateQueries({ queryKey: apiKeyKeys.list() }),
  });
}
