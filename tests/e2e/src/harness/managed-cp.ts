import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer as createHttpsServer, type Server as HttpsServer } from "node:https";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer as createTlsServer, type Server as TlsServer } from "node:tls";

import { etcdEndpoint } from "./etcd.js";

/**
 * A stand-in for the control plane a managed gateway talks to: dp-manager's
 * REST surface (`/dp/heartbeat`, `/dp/telemetry`) over HTTPS, and its etcd
 * endpoint as a TLS front for this fork's own etcd.
 *
 * Everything managed mode needs comes with it: a throwaway CA, the server
 * certificate both listeners present, and the gateway's client certificate
 * whose URI SANs carry the environment and data-plane ids (the shape the
 * control plane's issuer mints). Spread `managed` into `spawnApp`'s `extra`;
 * the gateway then reads its configuration under `<etcdPrefix>/<envId>/`.
 *
 * `telemetryStatus` decides how every `/dp/telemetry` POST is answered, so a
 * spec can take usage delivery down and bring it back. Every other path is
 * recorded in `calls` and answered `404`.
 */
export interface MockControlPlane {
  envId: string;
  /** The `managed` block for `spawnApp({ extra: { managed } })`. */
  managed: Record<string, unknown>;
  /** Status every `/dp/telemetry` POST is answered with; `200` to start. */
  telemetryStatus: number;
  /** Telemetry POSTs received, by the status they were answered with. */
  telemetryAnswers: number[];
  /** Every request path seen on the REST surface, in arrival order. */
  calls: string[];
  close(): Promise<void>;
}

function openssl(args: string[], cwd: string): void {
  execFileSync("openssl", args, { cwd, stdio: "pipe" });
}

/** Issue a leaf certificate under `ca` with the given extensions. */
function issue(dir: string, name: string, subject: string, extensions: string): void {
  openssl(
    [
      "req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
      "-keyout", `${name}.key`, "-out", `${name}.csr`, "-subj", subject,
    ],
    dir,
  );
  writeFileSync(join(dir, `${name}.ext`), extensions);
  openssl(
    [
      "x509", "-req", "-in", `${name}.csr`, "-CA", "ca.crt", "-CAkey", "ca.key",
      "-CAcreateserial", "-days", "2", "-out", `${name}.crt`, "-extfile", `${name}.ext`,
    ],
    dir,
  );
}

export async function startMockControlPlane(): Promise<MockControlPlane> {
  const envId = randomUUID();
  const dpId = randomUUID();
  const dir = mkdtempSync(join(tmpdir(), "aisix-e2e-cp-"));
  openssl(
    [
      "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
      "-keyout", "ca.key", "-out", "ca.crt", "-days", "2", "-subj", "/CN=aisix-e2e-ca",
      "-addext", "basicConstraints=critical,CA:TRUE",
      "-addext", "keyUsage=critical,keyCertSign,cRLSign",
    ],
    dir,
  );
  issue(
    dir,
    "server",
    "/CN=127.0.0.1",
    "basicConstraints=CA:FALSE\nsubjectAltName=IP:127.0.0.1,DNS:localhost\nextendedKeyUsage=serverAuth\n",
  );
  issue(
    dir,
    "client",
    `/CN=dp-${dpId}`,
    `basicConstraints=CA:FALSE\nsubjectAltName=URI:x-aisix://env/${envId},URI:x-aisix://dp/${dpId}\nextendedKeyUsage=clientAuth\n`,
  );
  const tlsMaterial = {
    key: readFileSync(join(dir, "server.key")),
    cert: readFileSync(join(dir, "server.crt")),
  };

  const cp: MockControlPlane = {
    envId,
    managed: {},
    telemetryStatus: 200,
    telemetryAnswers: [],
    calls: [],
    close: async () => {},
  };

  const rest: HttpsServer = createHttpsServer(tlsMaterial, (req, res) => {
    const path = (req.url ?? "").split("?")[0];
    cp.calls.push(path);
    req.resume();
    req.on("end", () => {
      if (req.method === "POST" && path === "/dp/heartbeat") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end("{}");
      } else if (req.method === "POST" && path === "/dp/telemetry") {
        const status = cp.telemetryStatus;
        cp.telemetryAnswers.push(status);
        res.writeHead(status, { "content-type": "application/json" });
        res.end(status < 300 ? '{"accepted":0,"rejected":0}' : '{"error":"unavailable"}');
      } else {
        res.writeHead(404);
        res.end();
      }
    });
  });
  rest.listen(0, "127.0.0.1");
  await once(rest, "listening");

  // gRPC over TLS in front of the plain etcd every spec shares.
  const target = new URL(etcdEndpoint());
  const sockets = new Set<Socket>();
  const etcdFront: TlsServer = createTlsServer({ ...tlsMaterial, ALPNProtocols: ["h2"] }, (client) => {
    const upstream = connect(Number(target.port), target.hostname);
    for (const s of [client, upstream]) {
      sockets.add(s);
      s.on("close", () => sockets.delete(s));
      s.on("error", () => s.destroy());
    }
    client.on("close", () => upstream.destroy());
    upstream.on("close", () => client.destroy());
    client.pipe(upstream);
    upstream.pipe(client);
  });
  etcdFront.listen(0, "127.0.0.1");
  await once(etcdFront, "listening");

  const port = (server: HttpsServer | TlsServer) => (server.address() as { port: number }).port;
  cp.managed = {
    enabled: true,
    cp_base_url: `https://127.0.0.1:${port(rest)}`,
    cp_etcd_endpoint: `127.0.0.1:${port(etcdFront)}`,
    cp_cert_file: join(dir, "client.crt"),
    cp_key_file: join(dir, "client.key"),
    cp_ca_file: join(dir, "ca.crt"),
    mtls_dir: join(dir, "mtls"),
    dp_id_file: join(dir, "mtls", "dp_id"),
  };
  cp.close = async () => {
    for (const s of sockets) s.destroy();
    rest.closeAllConnections();
    await Promise.all([
      new Promise<void>((resolve) => rest.close(() => resolve())),
      new Promise<void>((resolve) => etcdFront.close(() => resolve())),
    ]);
    rmSync(dir, { recursive: true, force: true });
  };
  return cp;
}
