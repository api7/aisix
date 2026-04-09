import { parseSseDataEvents } from './proxy.js';

interface ChatCompletionChunkLike {
  object: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  choices: Array<{
    index: number;
    finish_reason?: string | null;
    delta: {
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
  }>;
}

export const expectSdkErrorStatus = (err: unknown, expectedStatus: number) => {
  const status =
    typeof err === 'object' && err !== null && 'status' in err
      ? Number((err as { status: unknown }).status)
      : Number.NaN;

  expect(Number.isFinite(status)).toBe(true);
  expect(status).toBe(expectedStatus);
};

export const expectStreamHasDoneMarker = (sseBody: string) => {
  const events = parseSseDataEvents(sseBody);

  expect(events.length).toBeGreaterThan(0);
  expect(events).toContain('[DONE]');

  return events;
};

export const expectStreamStopsBeforeDone = (sseBody: string) => {
  const events = parseSseDataEvents(sseBody);

  expect(events.length).toBeGreaterThan(0);
  expect(events).not.toContain('[DONE]');

  return events;
};

export const expectParseableChatCompletionChunks = (sseBody: string) => {
  const events = expectStreamHasDoneMarker(sseBody).filter(
    (item) => item !== '[DONE]',
  );

  expect(events.length).toBeGreaterThan(0);

  const chunks = events.map(
    (item) => JSON.parse(item) as ChatCompletionChunkLike,
  );
  for (const chunk of chunks) {
    expect(chunk.object).toBe('chat.completion.chunk');
    expect(Array.isArray(chunk.choices)).toBe(true);
    if (chunk.choices.length > 0) {
      expect(typeof chunk.choices[0].index).toBe('number');
    } else {
      expect(chunk.usage).toBeDefined();
    }
  }

  return chunks;
};

export const expectStreamHasUsageChunk = (sseBody: string) => {
  const chunks = expectParseableChatCompletionChunks(sseBody);
  const usageChunks = chunks.filter((chunk) => chunk.usage !== undefined);

  expect(usageChunks.length).toBeGreaterThan(0);
  for (const chunk of usageChunks) {
    expect(typeof chunk.usage?.prompt_tokens).toBe('number');
    expect(typeof chunk.usage?.completion_tokens).toBe('number');
    expect(typeof chunk.usage?.total_tokens).toBe('number');
  }

  return usageChunks;
};

export const expectStreamHasToolCallDeltas = (sseBody: string) => {
  const chunks = expectParseableChatCompletionChunks(sseBody);
  const choiceChunks = chunks.flatMap((chunk) => chunk.choices);
  const toolCallDeltas = choiceChunks.flatMap(
    (choice) => choice.delta.tool_calls ?? [],
  );

  expect(toolCallDeltas.length).toBeGreaterThan(0);
  expect(toolCallDeltas.some((toolCall) => toolCall.id !== undefined)).toBe(
    true,
  );
  expect(
    toolCallDeltas.some((toolCall) => toolCall.function?.name !== undefined),
  ).toBe(true);
  expect(
    toolCallDeltas.some(
      (toolCall) =>
        typeof toolCall.function?.arguments === 'string' &&
        toolCall.function.arguments.length > 0,
    ),
  ).toBe(true);
  expect(
    choiceChunks.some((choice) => choice.finish_reason === 'tool_calls'),
  ).toBe(true);

  return { chunks, toolCallDeltas };
};
