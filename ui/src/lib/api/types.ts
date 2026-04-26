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

export interface Model {
  name: string;
  provider_id: string;
  model: string;
  timeout?: number;
  rate_limit?: RateLimit;
}

export const PROVIDER_TYPE_VARIANTS = [
  'openai',
  'openrouter',
  'azure',
  'anthropic',
  'gemini',
  'deepseek',
  'bedrock',
] as const;

export type ProviderType = (typeof PROVIDER_TYPE_VARIANTS)[number];

export interface ApiBaseProviderConfig {
  api_key: string;
  api_base?: string;
}

export interface AzureProviderConfig {
  api_key: string;
  api_base: string;
  api_version?: string;
}

export interface BedrockProviderConfig {
  region: string;
  access_key_id: string;
  secret_access_key: string;
  session_token?: string;
  endpoint?: string;
}

export type Provider =
  | {
      name: string;
      type: 'anthropic';
      config: ApiBaseProviderConfig;
    }
  | {
      name: string;
      type: 'azure';
      config: AzureProviderConfig;
    }
  | {
      name: string;
      type: 'deepseek';
      config: ApiBaseProviderConfig;
    }
  | {
      name: string;
      type: 'gemini';
      config: ApiBaseProviderConfig;
    }
  | {
      name: string;
      type: 'openai';
      config: ApiBaseProviderConfig;
    }
  | {
      name: string;
      type: 'openrouter';
      config: ApiBaseProviderConfig;
    }
  | {
      name: string;
      type: 'bedrock';
      config: BedrockProviderConfig;
    };

// ── ApiKey ────────────────────────────────────────────────────────────────────
export interface ApiKey {
  key: string;
  allowed_models: string[];
  rate_limit?: RateLimit;
}
