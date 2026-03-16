import OpenAI from 'openai';

import type { ChatMessage } from './state';

interface RunPlaygroundCompletionParams {
  apiKey: string;
  model: string;
  messages: ChatMessage[];
  paramsBody: Record<string, unknown> | null;
  stream: boolean;
  onStreamChunk?: (content: string) => void;
}

const PLAYGROUND_API_ROOT = '/playground';

function createPlaygroundClient(apiKey: string): OpenAI {
  return new OpenAI({
    apiKey,
    baseURL: new URL(PLAYGROUND_API_ROOT, window.location.origin).toString(),
    dangerouslyAllowBrowser: true,
  });
}

export async function runPlaygroundCompletion({
  apiKey,
  model,
  messages,
  paramsBody,
  stream,
  onStreamChunk,
}: RunPlaygroundCompletionParams): Promise<string> {
  const client = createPlaygroundClient(apiKey);
  const basePayload = {
    ...((paramsBody ?? {}) as Record<string, unknown>),
    model,
    messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
    stream,
  };

  if (!stream) {
    const result = await client.chat.completions.create(
      basePayload as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
    );
    return result.choices[0]?.message?.content ?? '';
  }

  const streamResult = await client.chat.completions.create(
    basePayload as OpenAI.Chat.ChatCompletionCreateParamsStreaming,
  );
  let content = '';

  for await (const chunk of streamResult) {
    const delta = chunk.choices[0]?.delta?.content ?? '';
    if (!delta) {
      continue;
    }

    content += delta;
    onStreamChunk?.(content);
  }

  return content;
}
