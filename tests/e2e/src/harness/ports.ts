import { createServer } from "node:net";

/**
 * Ask the OS for a free TCP port by binding to :0, reading the assigned
 * port, and closing the socket. There is a tiny race between close and
 * the caller binding, but in practice the test process grabs the port
 * fast enough that collisions are extremely rare.
 */
export async function pickFreePort(): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    const srv = createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const address = srv.address();
      if (!address || typeof address === "string") {
        reject(new Error("unexpected unix-socket address"));
        return;
      }
      const port = address.port;
      srv.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

/** Pick N free ports in sequence. */
export async function pickFreePorts(n: number): Promise<number[]> {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(await pickFreePort());
  return out;
}
