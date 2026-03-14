export type Role = 'system' | 'user' | 'assistant' | 'json';

export interface Message {
  id: string;
  role: Role;
  content: string;
}

export interface Params {
  stream: boolean;
  custom: boolean;
  json: string;
  max_tokens: string;
  temperature: string;
  top_p: string;
  n: string;
}

export interface ColumnState {
  id: string;
  apiKeyKey: string;
  modelKey: string;
  messages: Message[];
  isLoading: boolean;
  error?: string;
  parametersOpen: boolean;
  params: Params;
}

export type ModelItem = { key: string; value: { name: string; model: string } };
export type ApiKeyItem = { key: string; value: { key: string } };

export const ROLES: Role[] = ['system', 'user', 'assistant', 'json'];

export function makeMsgId(): string {
  return crypto.randomUUID();
}

export function defaultParams(): Params {
  return {
    stream: true,
    custom: false,
    json: JSON.stringify(
      { stream: true, max_tokens: 8192, temperature: 1, top_p: 1, n: 1 },
      null,
      2,
    ),
    max_tokens: '8192',
    temperature: '1',
    top_p: '1',
    n: '1',
  };
}

export function makeColumn(): ColumnState {
  return {
    id: crypto.randomUUID(),
    apiKeyKey: '',
    modelKey: '',
    messages: [
      {
        id: makeMsgId(),
        role: 'system',
        content: 'You are a helpful AI assistant.',
      },
    ],
    isLoading: false,
    parametersOpen: false,
    params: defaultParams(),
  };
}
