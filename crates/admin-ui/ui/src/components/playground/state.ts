import { defaultParams, type ColumnState } from './types';

export type ChatMessage = { role: string; content: string };
export const PLAYGROUND_COLUMNS_STORAGE_KEY = 'aisix-playground-columns-v1';

function isRole(
  value: unknown,
): value is 'system' | 'user' | 'assistant' | 'json' {
  return (
    value === 'system' ||
    value === 'user' ||
    value === 'assistant' ||
    value === 'json'
  );
}

export function parseStoredColumns(raw: string | null): ColumnState[] | null {
  if (!raw) return null;

  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return null;

    const defaults = defaultParams();

    return parsed
      .filter(
        (item): item is Record<string, unknown> =>
          !!item && typeof item === 'object',
      )
      .map((item) => {
        const params =
          item.params && typeof item.params === 'object'
            ? (item.params as Record<string, unknown>)
            : {};

        return {
          id:
            typeof item.id === 'string' && item.id.trim()
              ? item.id
              : crypto.randomUUID(),
          apiKeyKey: typeof item.apiKeyKey === 'string' ? item.apiKeyKey : '',
          modelKey: typeof item.modelKey === 'string' ? item.modelKey : '',
          messages: Array.isArray(item.messages)
            ? item.messages
                .filter(
                  (msg): msg is Record<string, unknown> =>
                    !!msg && typeof msg === 'object',
                )
                .map((msg) => ({
                  id:
                    typeof msg.id === 'string' && msg.id.trim()
                      ? msg.id
                      : crypto.randomUUID(),
                  role: isRole(msg.role) ? msg.role : 'user',
                  content: typeof msg.content === 'string' ? msg.content : '',
                }))
            : [],
          isLoading: false,
          error: undefined,
          parametersOpen: false,
          params: {
            stream: params.stream !== false,
            custom: !!params.custom,
            json: typeof params.json === 'string' ? params.json : defaults.json,
            max_tokens:
              typeof params.max_tokens === 'string'
                ? params.max_tokens
                : defaults.max_tokens,
            temperature:
              typeof params.temperature === 'string'
                ? params.temperature
                : defaults.temperature,
            top_p:
              typeof params.top_p === 'string' ? params.top_p : defaults.top_p,
            n: typeof params.n === 'string' ? params.n : defaults.n,
          },
        };
      })
      .filter((col) => col.messages.length > 0);
  } catch {
    return null;
  }
}

export function toApiMessages(column: ColumnState): ChatMessage[] {
  const output: ChatMessage[] = [];

  for (const msg of column.messages) {
    if (msg.role !== 'json') {
      output.push({ role: msg.role, content: msg.content });
      continue;
    }

    try {
      const parsed = JSON.parse(msg.content);
      if (Array.isArray(parsed)) {
        for (const item of parsed) {
          if (
            item &&
            typeof item === 'object' &&
            typeof item.role === 'string' &&
            typeof item.content === 'string'
          ) {
            output.push({ role: item.role, content: item.content });
          }
        }
      } else if (
        parsed &&
        typeof parsed === 'object' &&
        typeof parsed.role === 'string' &&
        typeof parsed.content === 'string'
      ) {
        output.push({ role: parsed.role, content: parsed.content });
      }
    } catch {
      output.push({ role: 'user', content: msg.content });
    }
  }

  return output;
}
