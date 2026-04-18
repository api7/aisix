// Shared types mirroring `aisix-core` entities and the
// `{prefix}/{kind}/{id}` resource-entry envelope returned by the admin
// API. Kept narrow on purpose — the UI only needs the fields it
// renders.

export interface RateLimit {
  rpm?: number | null;
  rpd?: number | null;
  tpm?: number | null;
  tpd?: number | null;
  concurrency?: number | null;
}

export interface ProviderConfig {
  api_key: string;
  api_base?: string | null;
}

export interface Model {
  name: string;
  model: string; // "<provider>/<upstream>"
  provider_config: ProviderConfig;
  rate_limit?: RateLimit | null;
  timeout_ms?: number | null;
}

export interface ApiKey {
  key: string;
  allowed_models: string[];
  rate_limit?: RateLimit | null;
}

export interface ResourceEntry<T> {
  id: string;
  value: T;
  revision: number;
}

// Spec §3 admin envelope.
export interface AdminError {
  error_msg: string;
}
