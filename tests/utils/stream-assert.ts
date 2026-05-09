import { parseSseDataEvents } from './proxy.js';

interface ToolCallDeltaLike {
  index: number;
  id?: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
}

interface ChatCompletionChunkLike {
  object: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    prompt_tokens_details?: {
      cached_tokens?: number;
      audio_tokens?: number;
    };
    completion_tokens_details?: {
      reasoning_tokens?: number;
      audio_tokens?: number;
    };
  };
  choices: Array<{
    index: number;
    finish_reason?: string | null;
    delta: {
      content?: string;
      tool_calls?: ToolCallDeltaLike[];
    };
  }>;
}

interface AggregatedToolCallLike {
  index: number;
  id?: string;
  type?: string;
  function: {
    name?: string;
    arguments: string;
  };
}

interface CompletionLike {
  choices: Array<{
    message?: {
      content?: string | null;
    };
  }>;
}

export interface ChatCompletionStreamSummary {
  chunks: ChatCompletionChunkLike[];
  text: string;
  usageChunks: ChatCompletionChunkLike[];
  finishReasons: string[];
  toolCallDeltas: ToolCallDeltaLike[];
  toolCalls: AggregatedToolCallLike[];
}

const chatCompletionEvents = (sseBody: string, requireDone = true) => {
  const events = parseSseDataEvents(sseBody);

  expect(events.length).toBeGreaterThan(0);

  if (requireDone) {
    expect(events).toContain('[DONE]');
  } else {
    expect(events).not.toContain('[DONE]');
  }

  return events.filter((item) => item !== '[DONE]');
};

const aggregateToolCalls = (chunks: ChatCompletionChunkLike[]) => {
  const aggregated = new Map<string, AggregatedToolCallLike>();

  for (const chunk of chunks) {
    for (const choice of chunk.choices) {
      for (const toolCall of choice.delta.tool_calls ?? []) {
        const key = `${choice.index}:${toolCall.index}`;
        const previous = aggregated.get(key) ?? {
          index: toolCall.index,
          function: { arguments: '' },
        };

        aggregated.set(key, {
          index: toolCall.index,
          id: toolCall.id ?? previous.id,
          type: toolCall.type ?? previous.type,
          function: {
            name: toolCall.function?.name ?? previous.function.name,
            arguments:
              previous.function.arguments +
              (toolCall.function?.arguments ?? ''),
          },
        });
      }
    }
  }

  return [...aggregated.values()];
};

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
  const events = chatCompletionEvents(sseBody);

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

export const summarizeChatCompletionStream = (
  sseBody: string,
): ChatCompletionStreamSummary => {
  const chunks = expectParseableChatCompletionChunks(sseBody);
  const usageChunks = chunks.filter((chunk) => chunk.usage !== undefined);
  const finishReasons = chunks.flatMap((chunk) =>
    chunk.choices
      .map((choice) => choice.finish_reason)
      .filter((finishReason): finishReason is string => Boolean(finishReason)),
  );
  const text = chunks
    .flatMap((chunk) => chunk.choices)
    .map((choice) => choice.delta.content)
    .filter((content): content is string => typeof content === 'string')
    .join('');
  const toolCallDeltas = chunks.flatMap((chunk) =>
    chunk.choices.flatMap((choice) => choice.delta.tool_calls ?? []),
  );

  return {
    chunks,
    text,
    usageChunks,
    finishReasons,
    toolCallDeltas,
    toolCalls: aggregateToolCalls(chunks),
  };
};

export const expectStreamMatchesAssistantText = (
  sseBody: string,
  expectedText: string,
) => {
  const summary = summarizeChatCompletionStream(sseBody);

  expect(summary.text).toBe(expectedText);

  return summary;
};

export const expectStreamMatchesCompletion = (
  sseBody: string,
  completion: CompletionLike,
) => {
  const expectedText = completion.choices[0]?.message?.content ?? '';

  expect(typeof expectedText).toBe('string');

  return expectStreamMatchesAssistantText(sseBody, expectedText);
};

export const expectStreamHasUsageChunk = (sseBody: string) => {
  const { usageChunks } = summarizeChatCompletionStream(sseBody);

  expect(usageChunks.length).toBeGreaterThan(0);
  for (const chunk of usageChunks) {
    expect(typeof chunk.usage?.prompt_tokens).toBe('number');
    expect(typeof chunk.usage?.completion_tokens).toBe('number');
    expect(typeof chunk.usage?.total_tokens).toBe('number');
  }

  return usageChunks;
};

export const expectStreamHasUsageOnlyChunk = (sseBody: string) => {
  const usageChunks = expectStreamHasUsageChunk(sseBody).filter(
    (chunk) => chunk.choices.length === 0,
  );

  expect(usageChunks.length).toBeGreaterThan(0);

  return usageChunks;
};

export const expectStreamRetainsTrailingContentAfterFinishReason = (
  sseBody: string,
  expectedText?: string,
) => {
  const summary = summarizeChatCompletionStream(sseBody);
  const firstFinishedChunkIndex = summary.chunks.findIndex((chunk) =>
    chunk.choices.some(
      (choice) =>
        choice.finish_reason !== null && choice.finish_reason !== undefined,
    ),
  );

  expect(firstFinishedChunkIndex).toBeGreaterThanOrEqual(0);

  const trailingText = summary.chunks
    .slice(firstFinishedChunkIndex + 1)
    .flatMap((chunk) => chunk.choices)
    .map((choice) => choice.delta.content)
    .filter((content): content is string => typeof content === 'string')
    .join('');

  expect(trailingText.length).toBeGreaterThan(0);

  if (expectedText !== undefined) {
    expect(summary.text).toBe(expectedText);
  }

  return summary;
};

export const expectStreamHasToolCallDeltas = (sseBody: string) => {
  const summary = summarizeChatCompletionStream(sseBody);

  expect(summary.toolCallDeltas.length).toBeGreaterThan(0);
  expect(
    summary.toolCallDeltas.some((toolCall) => toolCall.id !== undefined),
  ).toBe(true);
  expect(
    summary.toolCallDeltas.some(
      (toolCall) => toolCall.function?.name !== undefined,
    ),
  ).toBe(true);
  expect(
    summary.toolCallDeltas.some(
      (toolCall) =>
        typeof toolCall.function?.arguments === 'string' &&
        toolCall.function.arguments.length > 0,
    ),
  ).toBe(true);
  expect(summary.finishReasons).toContain('tool_calls');

  return summary;
};

export const expectStreamToolCallsFormJson = (sseBody: string) => {
  const summary = expectStreamHasToolCallDeltas(sseBody);

  const parsedToolCalls = summary.toolCalls.map((toolCall) => ({
    ...toolCall,
    parsedArguments: JSON.parse(toolCall.function.arguments) as unknown,
  }));

  return { ...summary, parsedToolCalls };
};
