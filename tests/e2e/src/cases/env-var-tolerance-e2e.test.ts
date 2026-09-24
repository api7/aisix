import { afterEach, describe, expect, test } from "vitest";
import { spawnApp, type SpawnedApp } from "../harness/index.js";

// E2E: an `AISIX_*` environment variable the gateway does not recognise
// must not stop it from starting.
//
// The gateway reads `AISIX_<SECTION>__<KEY>` as configuration overrides
// and its root config struct rejects unknown fields, so anything else
// prefixed `AISIX_` used to abort the boot with `unknown field`. Two
// environments produce such names without anyone asking for them:
//
//   - Kubernetes service links. Every pod in a namespace gets six or seven
//     variables per Service, named after the Service — so one Service
//     called `aisix-oss` is enough to make every gateway pod in that
//     namespace crash-loop before it opens a listener.
//   - The gateway's own `AISIX_CONFIG`, the documented env fallback for
//     `--config`. Setting it was fatal, which meant the fallback could
//     never be used.
//
// Both are driven here against the real binary because the failure was a
// process that exited, which no in-process test can observe.

// A Service named `aisix-oss` exposing 9090, as kubelet spells it.
const SERVICE_LINKS: Record<string, string> = {
  AISIX_OSS_SERVICE_HOST: "10.96.0.12",
  AISIX_OSS_SERVICE_PORT: "9090",
  AISIX_OSS_PORT: "tcp://10.96.0.12:9090",
  AISIX_OSS_PORT_9090_TCP: "tcp://10.96.0.12:9090",
  AISIX_OSS_PORT_9090_TCP_PROTO: "tcp",
  AISIX_OSS_PORT_9090_TCP_PORT: "9090",
  AISIX_OSS_PORT_9090_TCP_ADDR: "10.96.0.12",
};

// The file source keeps this spec off etcd: its subject is the boot
// itself, and nothing here needs a resource.
const RESOURCES = '_format_version: "1"\n';

describe("unrecognised AISIX_* environment variables", () => {
  let app: SpawnedApp | undefined;

  afterEach(async () => {
    await app?.exit();
    app = undefined;
  });

  test("Kubernetes service links do not stop the gateway from serving", async () => {
    // spawnApp resolves only once `/livez` answers, so reaching the
    // assertions at all is the "it started" half. Before the fix this
    // rejected with `exited early with code=1`.
    app = await spawnApp({ resourcesFile: RESOURCES, extraEnv: SERVICE_LINKS });

    const livez = await fetch(`${app.proxyUrl}/livez`);
    expect(livez.status).toBe(200);

    // An operator who never sees the log has a gateway silently ignoring
    // something they may believe they configured, so the line has to name
    // the variable and say what to do about it.
    const log = app.output();
    expect(log).toContain(
      "AISIX_OSS_PORT_9090_TCP_PROTO was not applied as a configuration override",
    );
    expect(log).toContain("enableServiceLinks: false");
  });

  test("AISIX_CONFIG starts the gateway without a --config argument", async () => {
    app = await spawnApp({ resourcesFile: RESOURCES, configViaEnv: true });

    const livez = await fetch(`${app.proxyUrl}/livez`);
    expect(livez.status).toBe(200);
    // The variable the binary acted on is its own, so nothing is warned
    // about. Asserted on the warning's own opening phrase, which the
    // other case proves is emitted when there IS something to report.
    expect(app.output()).not.toContain("was not applied as a configuration override");
  });
});
