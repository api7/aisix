import { randomUUID } from 'node:crypto';

import {
  MODELS_URL,
  PROVIDERS_URL,
  adminPost,
  adminPut,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import {
  type OpenAiMockUpstream,
  buildOpenAiProviderConfig,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import { proxyPost } from '../utils/proxy.js';
import { App } from '../utils/setup.js';

const ADMIN_KEY = 'test_admin_key_responses_proxy';
const AUTHORIZED_KEY = 'sk-proxy-responses-authorized';
const LIMITED_KEY = 'sk-proxy-responses-limited';
const UPSTREAM_API_KEY = 'upstream-key-responses-proxy';
const UPSTREAM_MODEL = 'test-model';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 1000));
};

const parseResponsesSseEvents = (sseBody: string) => {
  const trimmed = sseBody.trim();
  if (!trimmed) {
    return [] as Array<{ event?: string; data: string }>;
  }

  return trimmed.split(/\r?\n\r?\n/).map((block) => {
    const lines = block
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    return {
      event: lines.find((line) => line.startsWith('event: '))?.slice(7),
      data: lines
        .filter((line) => line.startsWith('data: '))
        .map((line) => line.slice(6))
        .join('\n'),
    };
  });
};

describe('proxy /v1/responses', () => {
  let server: App | undefined;
  let upstream: OpenAiMockUpstream | undefined;
  let mockModelName = '';
  let restrictedModelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
    upstream = await startOpenAiMockUpstream();
    const auth = bearerAuthHeader(ADMIN_KEY);

    mockModelName = `mock-responses-${randomUUID()}`;
    restrictedModelName = `mock-responses-restricted-${randomUUID()}`;
    const mockProviderId = `mock-responses-provider-${randomUUID()}`;
    const restrictedProviderId = `mock-responses-restricted-provider-${randomUUID()}`;

    const mockProviderResp = await adminPut(
      `${PROVIDERS_URL}/${mockProviderId}`,
      {
        name: mockProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(upstream.apiBase, UPSTREAM_API_KEY),
      },
      auth,
    );
    expect(mockProviderResp.status).toBe(201);

    const mockModelResp = await adminPost(
      MODELS_URL,
      {
        name: mockModelName,
        model: UPSTREAM_MODEL,
        provider_id: mockProviderId,
      },
      auth,
    );
    expect(mockModelResp.status).toBe(201);

    const restrictedProviderResp = await adminPut(
      `${PROVIDERS_URL}/${restrictedProviderId}`,
      {
        name: restrictedProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(upstream.apiBase, UPSTREAM_API_KEY),
      },
      auth,
    );
    expect(restrictedProviderResp.status).toBe(201);

    const restrictedModelResp = await adminPost(
      MODELS_URL,
      {
        name: restrictedModelName,
        model: UPSTREAM_MODEL,
        provider_id: restrictedProviderId,
      },
      auth,
    );
    expect(restrictedModelResp.status).toBe(201);

    const authorizedResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [mockModelName, restrictedModelName],
      },
      auth,
    );
    expect(authorizedResp.status).toBe(201);

    const limitedResp = await adminPost(
      '/apikeys',
      {
        key: LIMITED_KEY,
        allowed_models: [mockModelName],
      },
      auth,
    );
    expect(limitedResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('authorized upstream-backed model returns responses shape', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: mockModelName,
        input: 'hello from responses route',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('response');
    expect(resp.data.status).toBe('completed');
    expect(Array.isArray(resp.data.output)).toBe(true);
    expect(resp.data.output[0].type).toBe('message');
    expect(resp.data.output[0].content[0].type).toBe('output_text');
    expect(resp.data.output[0].content[0].text).toBe(
      'hello from mock upstream',
    );
    expect(resp.data.usage.input_tokens).toBe(10);
    expect(resp.data.usage.output_tokens).toBe(8);
    expect(resp.data.usage.total_tokens).toBe(18);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(recorded[0]?.headers.authorization).toBe(
      `Bearer ${UPSTREAM_API_KEY}`,
    );
    expect(
      (
        recorded[0]?.bodyJson as {
          model: string;
          messages: Array<{ content: string }>;
        }
      ).model,
    ).toBe(UPSTREAM_MODEL);
    expect(
      (
        recorded[0]?.bodyJson as {
          messages: Array<{ content: string }>;
        }
      ).messages[0]?.content,
    ).toBe('hello from responses route');
  });

  test('unauthorized model returns forbidden error', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: restrictedModelName,
        input: 'forbidden request',
      },
      LIMITED_KEY,
    );

    expect(resp.status).toBe(403);
    expect(resp.data.error.code).toBe('model_access_forbidden');
  });

  test('stream response emits responses event sequence without done marker', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: mockModelName,
        input: 'stream once',
        stream: true,
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');

    const events = parseResponsesSseEvents(String(resp.data));
    expect(events.length).toBeGreaterThan(0);
    expect(events.some((event) => event.data === '[DONE]')).toBe(false);
    expect(events[0]?.event).toBe('response.created');
    expect(
      events.some((event) => event.event === 'response.output_text.delta'),
    ).toBe(true);
    expect(events.at(-1)?.event).toBe('response.completed');

    const parsed = events.map((event) => ({
      event: event.event,
      data: JSON.parse(event.data) as { type: string },
    }));

    for (const event of parsed) {
      expect(event.data.type).toBe(event.event);
    }
  });

  test('previous_response_id replays session history through proxy gateway wiring', async () => {
    const firstResp = await proxyPost(
      '/v1/responses',
      {
        model: mockModelName,
        input: 'hello',
      },
      AUTHORIZED_KEY,
    );

    expect(firstResp.status).toBe(200);
    const firstResponseId = firstResp.data.id as string;

    const secondResp = await proxyPost(
      '/v1/responses',
      {
        model: mockModelName,
        input: 'how are you?',
        previous_response_id: firstResponseId,
      },
      AUTHORIZED_KEY,
    );

    expect(secondResp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(2);

    const secondBody = recorded[1]?.bodyJson as {
      messages: Array<{ role: string; content: string }>;
    };

    expect(secondBody.messages[0]?.role).toBe('user');
    expect(secondBody.messages[0]?.content).toBe('hello');
    expect(secondBody.messages[1]?.role).toBe('assistant');
    expect(secondBody.messages[1]?.content).toBe('hello from mock upstream');
    expect(secondBody.messages[2]?.role).toBe('user');
    expect(secondBody.messages[2]?.content).toBe('how are you?');
  });

  test('missing previous_response_id returns validation before upstream dispatch', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: mockModelName,
        input: 'hello',
        previous_response_id: 'resp_missing',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(resp.data.error.message).toContain('previous_response_not_found');
    expect(upstream?.takeRecordedRequests() ?? []).toHaveLength(0);
  });
});
