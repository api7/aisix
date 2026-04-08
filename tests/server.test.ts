import path from 'node:path';

import { client } from './utils/http.js';
import {
  App,
  ERR_UNEXPECTED_EARLY_EXIT,
  defaultConfig,
  randomPort,
  tlsSkipVerify,
} from './utils/setup.js';

const INVALID_LISTEN_ADDR = 'invalid-listen-addr';
const ERR_PORT_IN_USE = 'Address already in use';

const CERT_FILE = path.resolve('fixtures/tls/server.cer');
const KEY_FILE = path.resolve('fixtures/tls/server.key');
const NON_EXISTENT_CERT_FILE = path.resolve('fixtures/tls/server-ne.cer');
const NON_EXISTENT_KEY_FILE = path.resolve('fixtures/tls/server-ne.key');

describe('proxy server', () => {
  const expectedStatus = 401;

  let server: App | undefined;

  afterEach(async () => await server?.exit());

  test('listen http (default addr)', async () => {
    server = await (await App.spawn()).waitForReady();

    const resp = await client.get(`http://127.0.0.1:3000/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('listen http (specify addr)', async () => {
    const port = randomPort();
    server = await (
      await App.spawn(
        defaultConfig({ server: { proxy: { listen: `127.0.0.1:${port}` } } }),
      )
    ).waitForReady(port);

    const resp = await client.get(`http://127.0.0.1:${port}/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('empty listen', async () => {
    server = await (
      await App.spawn(
        defaultConfig({ server: { proxy: { listen: undefined } } }),
      )
    ).waitForReady(3000);

    const resp = await client.get(`http://127.0.0.1:3000/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('invalid listen', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({ server: { proxy: { listen: INVALID_LISTEN_ADDR } } }),
        1000,
      ),
    ).rejects.toThrow(ERR_UNEXPECTED_EARLY_EXIT);
  });

  test('port in use', async () => {
    server = await (
      await App.spawn(
        defaultConfig({
          // avoid conflict with admin server
          server: { admin: { listen: '127.0.0.1:30000' } },
        }),
      )
    ).waitForReady(3000);

    const resp = await client.get(`http://127.0.0.1:3000/`);
    expect(resp.status).toBe(expectedStatus);

    await expect(
      App.spawn(
        defaultConfig({
          server: { admin: { listen: '127.0.0.1:30001' } },
        }),
        1000,
      ),
    ).rejects.toThrow(ERR_PORT_IN_USE);
  });

  test('tls enabled', async () => {
    server = await (
      await App.spawn(
        defaultConfig({
          server: {
            proxy: {
              tls: {
                enabled: true,
                cert_file: CERT_FILE,
                key_file: KEY_FILE,
              },
            },
          },
        }),
      )
    ).waitForReady('https://127.0.0.1:3000');

    const resp = await client.get(`https://127.0.0.1:3000/`, {
      httpsAgent: tlsSkipVerify,
    });
    expect(resp.status).toBe(expectedStatus);
  });

  test('invalid tls config (missing key_file)', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: {
            proxy: {
              tls: {
                enabled: true,
                cert_file: CERT_FILE,
              },
            },
          },
        }),
        1000,
      ),
    ).rejects.toThrow('key_file is required when TLS is enabled');
  });

  test('invalid tls config (file not exist)', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: {
            proxy: {
              tls: {
                enabled: true,
                cert_file: NON_EXISTENT_CERT_FILE,
                key_file: NON_EXISTENT_KEY_FILE,
              },
            },
          },
        }),
        1000,
      ),
    ).rejects.toThrow('does not exist');
  });

  test('invalid tls config (file not cert)', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: {
            proxy: {
              tls: {
                enabled: true,
                cert_file: KEY_FILE,
                key_file: KEY_FILE,
              },
            },
          },
        }),
        1000,
      ),
    ).rejects.toThrow('Expecting: CERTIFICATE');
  });
});

describe('admin server', () => {
  const expectedStatus = 404;

  let server: App | undefined;

  afterEach(async () => await server?.exit());

  test('listen http (default addr)', async () => {
    server = await (await App.spawn()).waitForReady(3001);

    const resp = await client.get(`http://127.0.0.1:3001/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('listen http (specify addr)', async () => {
    const port = randomPort();
    server = await (
      await App.spawn(
        defaultConfig({
          server: { admin: { listen: `127.0.0.1:${port}` } },
        }),
      )
    ).waitForReady(port);

    const resp = await client.get(`http://127.0.0.1:${port}/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('empty listen', async () => {
    server = await (
      await App.spawn(
        defaultConfig({ server: { admin: { listen: undefined } } }),
      )
    ).waitForReady(3001);

    const resp = await client.get(`http://127.0.0.1:3001/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('invalid listen', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: { admin: { listen: INVALID_LISTEN_ADDR } },
        }),
        1000,
      ),
    ).rejects.toThrow(ERR_UNEXPECTED_EARLY_EXIT);
  });

  test('port in use', async () => {
    server = await (
      await App.spawn(
        defaultConfig(
          // avoid conflict with proxy server
          { server: { proxy: { listen: '127.0.0.1:30000' } } },
        ),
      )
    ).waitForReady(3001);

    const resp = await client.get(`http://127.0.0.1:3001/`);
    expect(resp.status).toBe(expectedStatus);

    await expect(
      App.spawn(
        defaultConfig({ server: { proxy: { listen: '127.0.0.1:30001' } } }),
        1000,
      ),
    ).rejects.toThrow(ERR_PORT_IN_USE);
  });

  test('tls enabled', async () => {
    server = await (
      await App.spawn(
        defaultConfig({
          server: {
            admin: {
              tls: {
                enabled: true,
                cert_file: CERT_FILE,
                key_file: KEY_FILE,
              },
            },
          },
        }),
      )
    ).waitForReady('https://127.0.0.1:3001');

    const resp = await client.get(`https://127.0.0.1:3001/`, {
      httpsAgent: tlsSkipVerify,
    });
    expect(resp.status).toBe(expectedStatus);
  });

  test('invalid tls config (missing key_file)', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: {
            admin: {
              tls: {
                enabled: true,
                cert_file: CERT_FILE,
              },
            },
          },
        }),
        1000,
      ),
    ).rejects.toThrow('key_file is required when TLS is enabled');
  });

  test('invalid tls config (file not exist)', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: {
            admin: {
              tls: {
                enabled: true,
                cert_file: NON_EXISTENT_CERT_FILE,
                key_file: NON_EXISTENT_KEY_FILE,
              },
            },
          },
        }),
        1000,
      ),
    ).rejects.toThrow('does not exist');
  });

  test('invalid tls config (file not cert)', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          server: {
            admin: {
              tls: {
                enabled: true,
                cert_file: KEY_FILE,
                key_file: KEY_FILE,
              },
            },
          },
        }),
        1000,
      ),
    ).rejects.toThrow('Expecting: CERTIFICATE');
  });
});
