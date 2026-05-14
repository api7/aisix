import { buildOpenAiProviderModel } from '../../utils/mock-upstream.js';
import { proxyPost } from '../../utils/proxy.js';
import {
  type RegexGuardrailFixture,
  setupOpenAiRegexGuardrailFixture,
} from './shared.js';

const ADMIN_KEY = 'test_admin_key_guardrail_messages';
const AUTHORIZED_KEY = 'sk-proxy-guardrail-messages';
const UPSTREAM_API_KEY = 'upstream-key-guardrail-messages';
const UPSTREAM_MODEL = 'test-model';

describe('proxy guardrail /v1/messages', () => {
  let fixture: RegexGuardrailFixture | undefined;

  beforeEach(async () => {
    fixture = await setupOpenAiRegexGuardrailFixture({
      adminKey: ADMIN_KEY,
      authorizedKey: AUTHORIZED_KEY,
      upstreamApiKey: UPSTREAM_API_KEY,
      upstreamModel: UPSTREAM_MODEL,
      modelPrefix: 'mock-messages-guardrail',
      buildModel: buildOpenAiProviderModel,
    });
  }, 30_000);

  afterEach(async () => {
    await fixture?.close();
  });

  test('input regex guardrail blocks messages request before upstream call', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: fixture?.inputGuardedModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'my secret token is 12345' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.type).toBe('error');
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(resp.data.error.message).toBe('Invalid request');
    expect(typeof resp.data.request_id).toBe('string');

    const recorded = fixture?.upstream.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(0);
  });

  test('input regex guardrail allows safe messages request through to upstream', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: fixture?.inputGuardedModelName,
        max_tokens: 256,
        messages: [
          { role: 'user', content: 'safe request through regex guardrail' },
        ],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.type).toBe('message');
    expect(resp.data.content[0].text).toBe('hello from mock upstream');

    const recorded = fixture?.upstream.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (
        recorded[0]?.bodyJson as {
          messages: Array<{ content: string }>;
        }
      ).messages[0]?.content,
    ).toBe('safe request through regex guardrail');
  });

  test('output regex guardrail blocks matched messages response', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: fixture?.outputGuardedModelName,
        max_tokens: 256,
        messages: [
          { role: 'user', content: 'safe prompt for output guardrail' },
        ],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.type).toBe('error');
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(resp.data.error.message).toBe('Invalid request');
    expect(typeof resp.data.request_id).toBe('string');

    const recorded = fixture?.upstream.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (
        recorded[0]?.bodyJson as {
          messages: Array<{ content: string }>;
        }
      ).messages[0]?.content,
    ).toBe('safe prompt for output guardrail');
  });
});
