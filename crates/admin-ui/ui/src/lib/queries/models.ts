import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { useAdminKey } from '@/hooks/use-admin-key';
import { ApiClientError, modelsApi } from '@/lib/api/client';
import type { Model } from '@/lib/api/types';

export const modelKeys = {
  all: ['models'] as const,
  list: () => [...modelKeys.all, 'list'] as const,
  detail: (id: string) => [...modelKeys.all, 'detail', id] as const,
};

export function useModels() {
  const { key, openModal } = useAdminKey();
  return useQuery({
    queryKey: modelKeys.list(),
    queryFn: () => modelsApi.list(key!),
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

export function useModel(id: string) {
  const { key, openModal } = useAdminKey();
  return useQuery({
    queryKey: modelKeys.detail(id),
    queryFn: () => modelsApi.get(key!, id),
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

export function useCreateModel() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (data: Model) => modelsApi.create(key!, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: modelKeys.list() }),
  });
}

export function useUpdateModel(id: string) {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (data: Model) => modelsApi.update(key!, id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: modelKeys.list() });
      qc.invalidateQueries({ queryKey: modelKeys.detail(id) });
    },
  });
}

export function useDeleteModel() {
  const qc = useQueryClient();
  const { key } = useAdminKey();
  return useMutation({
    mutationFn: (id: string) => modelsApi.delete(key!, id),
    onSuccess: () => qc.invalidateQueries({ queryKey: modelKeys.list() }),
  });
}
