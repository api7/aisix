import { randomUUID } from 'node:crypto';

import {
  POLICIES_URL,
  adminDelete,
  adminGet,
  adminPost,
  adminPut,
  bearerAuthHeader,
  extractIdFromStorageKey,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import { etcdPutJson } from '../utils/etcd.js';
import { App } from '../utils/setup.js';

const ADMIN_KEY = 'test_admin_key';

const seedRegexGuardrail = async (etcdPrefix: string, guardrailId: string) => {
  await etcdPutJson(etcdPrefix, `/guardrails/${guardrailId}`, {
    name: `${guardrailId}-name`,
    type: 'regex',
    config: {
      pattern: 'blocked phrase',
      block_reason: 'blocked by policy admin test guardrail',
    },
  });
};

const buildPolicyBody = (name: string, guardrailId: string, when = 'true') => ({
  name,
  when,
  actions: [
    {
      type: 'guardrail',
      config: {
        guardrail_ids: [guardrailId],
      },
    },
  ],
});

describe('admin policies', () => {
  let server: App | undefined;
  let etcdPrefix = '';

  beforeEach(async () => {
    etcdPrefix = `/ai-admin-${randomUUID()}`;
    server = await startIsolatedAdminApp(ADMIN_KEY, etcdPrefix);
  });

  afterEach(async () => await server?.exit());

  test('test_crud', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const guardrailId = `policy-crud-guardrail-${randomUUID()}`;

    await seedRegexGuardrail(etcdPrefix, guardrailId);

    const listBefore = await adminGet(POLICIES_URL, auth);
    expect(listBefore.status).toBe(200);
    expect(listBefore.data.total).toBe(0);

    const createResp = await adminPost(
      POLICIES_URL,
      buildPolicyBody('test_policy', guardrailId),
      auth,
    );
    expect(createResp.status).toBe(201);
    expect(createResp.data.value.enabled).toBe(true);
    expect(createResp.data.value.priority).toBe(0);
    expect(createResp.data.value.actions[0].config.stages).toStrictEqual([
      'input',
      'output',
    ]);
    const id = extractIdFromStorageKey(createResp.data.key);

    const listAfterCreate = await adminGet(POLICIES_URL, auth);
    expect(listAfterCreate.status).toBe(200);
    expect(listAfterCreate.data.total).toBe(1);

    const updateResp = await adminPut(
      `${POLICIES_URL}/${id}`,
      {
        name: 'updated_policy',
        enabled: false,
        priority: 42,
        when: "route.format == 'chat_completions'",
        actions: [
          {
            type: 'guardrail',
            config: {
              stages: ['input'],
              guardrail_ids: [guardrailId],
            },
          },
        ],
      },
      auth,
    );
    expect(updateResp.status).toBe(200);
    expect(updateResp.data.value.name).toBe('updated_policy');
    expect(updateResp.data.value.enabled).toBe(false);
    expect(updateResp.data.value.priority).toBe(42);
    expect(updateResp.data.value.actions[0].config.stages).toStrictEqual([
      'input',
    ]);

    const getResp = await adminGet(`${POLICIES_URL}/${id}`, auth);
    expect(getResp.status).toBe(200);
    expect(getResp.data.value.name).toBe('updated_policy');

    const deleteResp = await adminDelete(`${POLICIES_URL}/${id}`, auth);
    expect(deleteResp.status).toBe(200);
    expect(deleteResp.data.deleted).toBe(1);

    const listAfterDelete = await adminGet(POLICIES_URL, auth);
    expect(listAfterDelete.status).toBe(200);
    expect(listAfterDelete.data.total).toBe(0);
  });

  test('test_put_status_codes', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const guardrailId = `policy-put-guardrail-${randomUUID()}`;

    await seedRegexGuardrail(etcdPrefix, guardrailId);

    const body = buildPolicyBody('put_policy', guardrailId);

    const firstPut = await adminPut(
      `${POLICIES_URL}/put-policy-fixed-id`,
      body,
      auth,
    );
    expect(firstPut.status).toBe(201);

    const secondPut = await adminPut(
      `${POLICIES_URL}/put-policy-fixed-id`,
      body,
      auth,
    );
    expect(secondPut.status).toBe(200);
  });

  test('test_put_duplicate_name_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const guardrailId = `policy-dup-guardrail-${randomUUID()}`;

    await seedRegexGuardrail(etcdPrefix, guardrailId);

    const putA = await adminPut(
      `${POLICIES_URL}/put-dup-policy-a`,
      buildPolicyBody('put-dup-name-a', guardrailId),
      auth,
    );
    expect(putA.status).toBe(201);

    const putB = await adminPut(
      `${POLICIES_URL}/put-dup-policy-b`,
      buildPolicyBody('put-dup-name-b', guardrailId),
      auth,
    );
    expect(putB.status).toBe(201);

    const putDup = await adminPut(
      `${POLICIES_URL}/put-dup-policy-b`,
      buildPolicyBody('put-dup-name-a', guardrailId),
      auth,
    );
    expect(putDup.status).toBe(400);
    expect(putDup.data.error_msg).toBe('Policy name already exists');
  });

  test('test_missing_guardrail_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const missingGuardrailId = `missing-guardrail-${randomUUID()}`;

    const createResp = await adminPost(
      POLICIES_URL,
      buildPolicyBody('missing_guardrail_policy', missingGuardrailId),
      auth,
    );
    expect(createResp.status).toBe(400);
    expect(createResp.data.error_msg).toBe(
      `Guardrail with ID ${missingGuardrailId} not found`,
    );
  });

  test('test_invalid_cel_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const guardrailId = `policy-invalid-cel-guardrail-${randomUUID()}`;

    await seedRegexGuardrail(etcdPrefix, guardrailId);

    const putResp = await adminPut(
      `${POLICIES_URL}/invalid-cel-policy`,
      buildPolicyBody('invalid_cel_policy', guardrailId, 'route.format =='),
      auth,
    );
    expect(putResp.status).toBe(400);
    expect(putResp.data.error_msg).toContain(
      'CEL validation error on policy "invalid-cel-policy"',
    );
  });
});
