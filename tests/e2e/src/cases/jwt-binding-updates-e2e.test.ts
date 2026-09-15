import { createHash, randomUUID } from "node:crypto";
import { expect, test } from "vitest";
import {
  agentClaims, EtcdClient, ProxyClient, SeedClient, spawnApp,
  startMockIdp, startOpenAiUpstream, waitConfigPropagation,
  type MockIdp, type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";

test("warmed JWT bindings follow key edits, revocation, ambiguity and claim fallback", async (ctx) => {
  const etcd = new EtcdClient();
  if (!(await etcd.ping())) { ctx.skip(); return; }
  let app: SpawnedApp | undefined;
  let idp: MockIdp | undefined;
  let upstream: OpenAiUpstream | undefined;
  const errors: unknown[] = [];
  try {
    upstream = await startOpenAiUpstream();
    idp = await startMockIdp();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({ display_name: "jwt-updates-pk", api_key: "sk-mock", api_base: `${upstream.baseUrl}/v1` });
    for (const name of ["direct-model", "fallback-model"]) {
      await seed.createModel({ display_name: name, provider: "openai", model_name: "gpt-4o-mini", provider_key_id: pk.id });
    }
    await seed.createOidcProvider({ name: "provider", issuer: idp.url, audiences: ["aisix-gateway"], jwks_uri: idp.jwksUrl });
    const fallback = await seed.createApiKey({ key_hash: createHash("sha256").update("fallback").digest("hex"), allowed_models: ["fallback-model"] });
    await seed.createClaimMapping({ name: "fallback", jwt_provider: "provider", priority: 1,
      match: [{ claim: "department", op: "exact", values: ["test"] }], resolve: { api_key_id: fallback.id } });
    const binding = await seed.createApiKey({
      key_hash: createHash("sha256").update("bound").digest("hex"), allowed_models: ["direct-model"],
      jwt_provider: "provider", jwt_subject: "subject",
    });
    // An ordinary final key proves each entire update has landed without exercising JWT resolution.
    const applied = async () => {
      const secret = `sk-ready-${randomUUID()}`;
      await seed.update("api_keys", marker, { key_hash: createHash("sha256").update(secret).digest("hex"), allowed_models: [] });
      const proxy = new ProxyClient(app!.proxyUrl, secret);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
    };
    const marker = randomUUID();
    const token = (subject = "subject") => idp!.sign(agentClaims(idp!.url, { sub: subject, department: "test" }));
    const request = async (model: string, bearer = token()) => {
      const response = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST", headers: { authorization: `Bearer ${bearer}`, "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "binding update" }] }),
      });
      return { status: response.status, body: await response.json() as { error?: { code?: string } } };
    };
    await applied();
    for (let i = 0; i < 3; i++) expect((await request("direct-model")).status).toBe(200);
    expect((await request("fallback-model")).status).toBe(403);

    await seed.update("api_keys", binding.id, { ...binding.value, disabled: true });
    await applied();
    expect(await request("fallback-model")).toMatchObject({ status: 401, body: { error: { code: "api_key_disabled" } } });
    await seed.update("api_keys", binding.id, { ...binding.value, expires_at: "2000-01-01T00:00:00Z" });
    await applied();
    expect(await request("fallback-model")).toMatchObject({ status: 401, body: { error: { code: "api_key_expired" } } });
    await seed.update("api_keys", binding.id, binding.value);
    await applied();
    expect((await request("direct-model")).status).toBe(200);

    const duplicate = await seed.createApiKey({ ...binding.value,
      key_hash: createHash("sha256").update("duplicate").digest("hex"), disabled: true });
    await applied();
    expect(await request("fallback-model")).toMatchObject({ status: 401, body: { error: { code: "jwt_identity_unmapped" } } });
    await seed.delete("api_keys", duplicate.id);
    await applied();
    expect((await request("direct-model")).status).toBe(200);

    await seed.update("api_keys", binding.id, { ...binding.value, jwt_subject: "renamed-subject" });
    await applied();
    expect((await request("fallback-model")).status).toBe(200);
    expect((await request("direct-model")).status).toBe(403);
    expect((await request("direct-model", token("renamed-subject"))).status).toBe(200);
    await seed.delete("api_keys", binding.id);
    await applied();
    expect((await request("fallback-model", token("renamed-subject"))).status).toBe(200);
    await seed.update("api_keys", binding.id, binding.value);
    await applied();
    expect((await request("direct-model")).status).toBe(200);
  } catch (error) {
    errors.push(error);
  }
  const cleanup = await Promise.allSettled([
    app?.exit(), app && etcd.deletePrefix(app.etcdPrefix), upstream?.close(), idp?.close(),
  ]);
  for (const result of cleanup) {
    if (result.status === "rejected") errors.push(result.reason);
  }
  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, "JWT binding E2E failure");
});
