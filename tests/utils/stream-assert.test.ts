import {
  type OpenAiMockStreamEvent,
  buildOpenAiToolCallStreamEvents,
  buildOpenAiTrailingContentAfterFinishReasonStreamEvents,
} from './mock-upstream.js';
import {
  expectStreamHasUsageOnlyChunk,
  expectStreamRetainsTrailingContentAfterFinishReason,
  expectStreamToolCallsFormJson,
} from './stream-assert.js';

const toSseBody = (events: OpenAiMockStreamEvent[]) =>
  events
    .map((event) => {
      if (typeof event === 'string') {
        return `data: ${event}`;
      }

      if ('data' in event) {
        const lines: string[] = [];
        if (event.event) {
          lines.push(`event: ${event.event}`);
        }

        const payload =
          typeof event.data === 'string'
            ? event.data
            : JSON.stringify(event.data);
        lines.push(`data: ${payload}`);

        return lines.join('\n');
      }

      return `data: ${JSON.stringify(event)}`;
    })
    .join('\n\n');

describe('stream assert helpers', () => {
  test('detects trailing content that arrives after the first finish_reason chunk', () => {
    const sseBody = toSseBody(
      buildOpenAiTrailingContentAfterFinishReasonStreamEvents('test-model'),
    );

    const summary = expectStreamRetainsTrailingContentAfterFinishReason(
      sseBody,
      'hello from trailing delta',
    );
    const usageOnlyChunks = expectStreamHasUsageOnlyChunk(sseBody);

    expect(summary.finishReasons).toContain('stop');
    expect(usageOnlyChunks).toHaveLength(1);
  });

  test('aggregates fragmented tool call arguments into valid json', () => {
    const sseBody = toSseBody(buildOpenAiToolCallStreamEvents('test-model'));

    const { parsedToolCalls } = expectStreamToolCallsFormJson(sseBody);

    expect(parsedToolCalls).toHaveLength(1);
    expect(parsedToolCalls[0]?.id).toBe('call_weather_1');
    expect(parsedToolCalls[0]?.function.name).toBe('get_weather');
    expect(parsedToolCalls[0]?.parsedArguments).toEqual({ city: 'Shanghai' });
  });
});
