import { client } from './http.js';
import { startOpenAiMockUpstream } from './mock-upstream.js';

describe('OpenAiMockUpstream scripted SSE support', () => {
  test('emits scripted frames with explicit event lines and custom payload strings', async () => {
    const upstream = await startOpenAiMockUpstream({
      streamEvents: [
        {
          event: 'mock.chunk',
          data: {
            id: 'chatcmpl-scripted-e2e-mock',
            object: 'chat.completion.chunk',
            created: 1,
            model: 'scripted-model',
            choices: [
              {
                index: 0,
                delta: { role: 'assistant', content: 'hello' },
                finish_reason: null,
              },
            ],
          },
        },
        { event: 'mock.marker', data: 'custom-marker' },
        '[DONE]',
      ],
    });

    try {
      const resp = await client.post(
        `${upstream.apiBase}/chat/completions`,
        {
          model: 'scripted-model',
          stream: true,
          messages: [{ role: 'user', content: 'hello upstream' }],
        },
        { responseType: 'text' },
      );

      expect(resp.status).toBe(200);
      expect(String(resp.headers['content-type'])).toContain(
        'text/event-stream',
      );
      expect(String(resp.data)).toContain('event: mock.chunk');
      expect(String(resp.data)).toContain('event: mock.marker');
      expect(String(resp.data)).toContain('data: custom-marker');
      expect(String(resp.data)).toContain('data: [DONE]');

      const recorded = upstream.takeRecordedRequests();
      expect(recorded).toHaveLength(1);
      expect(
        (recorded[0]?.bodyJson as { messages: Array<{ content: string }> })
          .messages[0]?.content,
      ).toBe('hello upstream');
    } finally {
      await upstream.close();
    }
  });
});
