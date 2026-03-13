// ── Shared response wrappers ──────────────────────────────────────────────────
export interface ListResponse<T> {
  total: number;
  list: Array<ItemResponse<T>>;
}

export interface ItemResponse<T> {
  key: string;
  value: T;
  created_index?: number;
  modified_index?: number;
}

export interface DeleteResponse {
  deleted: number;
  key: string;
}

export interface ApiError {
  error_msg: string;
}

// ── RateLimit ─────────────────────────────────────────────────────────────────
export interface RateLimit {
  tpm?: number;
  tpd?: number;
  rpm?: number;
  rpd?: number;
  concurrency?: number;
}

// ── Model ─────────────────────────────────────────────────────────────────────
export interface ProviderConfig {
  api_key?: string;
  api_base?: string;
  [key: string]: unknown;
}

export interface Model {
  name: string;
  /** Format: provider/model-name */
  model: string;
  provider_config: ProviderConfig;
  timeout?: number;
  rate_limit?: RateLimit;
}

// ── ApiKey ────────────────────────────────────────────────────────────────────
export interface ApiKey {
  key: string;
  allowed_models: string[];
  rate_limit?: RateLimit;
}
