import { proxyPost } from '../../utils/proxy.js';
import {
  type RegexGuardrailFixture,
  setupOpenAiRegexGuardrailFixture,
} from './shared.js';

const ADMIN_KEY = 'test_admin_key_guardrail_responses';
const AUTHORIZED_KEY = 'sk-proxy-guardrail-responses';
const UPSTREAM_API_KEY = 'upstream-key-guardrail-responses';
const UPSTREAM_MODEL = 'test-model';

describe('proxy guardrail /v1/responses', () => {
  let fixture: RegexGuardrailFixture | undefined;

  beforeEach(async () => {
    fixture = await setupOpenAiRegexGuardrailFixture({
      adminKey: ADMIN_KEY,
      authorizedKey: AUTHORIZED_KEY,
      upstreamApiKey: UPSTREAM_API_KEY,
      upstreamModel: UPSTREAM_MODEL,
      modelPrefix: 'mock-responses-guardrail',
    });
  }, 30_000);

  afterEach(async () => {
    await fixture?.close();
  });

  test('input regex guardrail blocks responses request before upstream call', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: fixture?.inputGuardedModelName,
        input: 'my secret token is 12345',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('gateway_error');
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(resp.data.error.message).toContain('guardrail regex blocked input');
    expect(resp.data.error.message).toContain(
      'blocked by regex input guardrail',
    );

    const recorded = fixture?.upstream.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(0);
  });

  test('input regex guardrail allows safe responses request through to upstream', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: fixture?.inputGuardedModelName,
        input: 'safe request through regex guardrail',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('response');
    expect(resp.data.output[0].content[0].text).toBe(
      'hello from mock upstream',
    );

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

  test('output regex guardrail blocks matched responses output', async () => {
    const resp = await proxyPost(
      '/v1/responses',
      {
        model: fixture?.outputGuardedModelName,
        input: 'safe prompt for output guardrail',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('gateway_error');
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(resp.data.error.message).toContain('guardrail regex blocked output');
    expect(resp.data.error.message).toContain(
      'blocked by regex output guardrail',
    );

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
