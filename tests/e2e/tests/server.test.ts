import { spawn } from 'child_process';

import { client } from '../utils/http.js';
import {
  App,
  AppConfig,
  ERR_UNEXPECTED_EARLY_EXIT,
  ERR_UNEXPECTED_EXIT,
  defaultConfig,
  randomPort,
} from '../utils/setup.js';

const INVALID_LISTEN_ADDR = 'invalid-listen-addr';

describe('proxy server', () => {
  const expectedStatus = 401;

  let server: App | undefined;

  afterEach(() => server?.exit());

  test('listen http (default addr)', async () => {
    server = await (await App.spawn()).waitForReady();

    const resp = await client.get(`http://localhost:3000/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('listen http (specify addr)', async () => {
    const port = randomPort();
    server = await (
      await App.spawn(defaultConfig({ listen: `127.0.0.1:${port}` }))
    ).waitForReady(port);

    const resp = await client.get(`http://localhost:${port}/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('empty listen', async () => {
    server = await (
      await App.spawn(defaultConfig({ listen: undefined }))
    ).waitForReady(3000);

    const resp = await client.get(`http://localhost:3000/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('invalid listen', async () => {
    server = undefined;

    await expect(
      App.spawn(defaultConfig({ listen: INVALID_LISTEN_ADDR }), 1000),
    ).rejects.toThrow(ERR_UNEXPECTED_EARLY_EXIT);
  });

  test('port in use', async () => {
    server = await (
      await App.spawn(
        defaultConfig({
          // avoid conflict with admin server
          deployment: { admin: { listen: '127.0.0.1:30000' } },
        }),
      )
    ).waitForReady(3000);

    const resp = await client.get(`http://localhost:3000/`);
    expect(resp.status).toBe(expectedStatus);

    await expect(
      App.spawn(
        defaultConfig({
          deployment: { admin: { listen: '127.0.0.1:30001' } },
        }),
        1000,
      ),
    ).rejects.toThrow(ERR_UNEXPECTED_EARLY_EXIT);
  });
});

describe('admin server', () => {
  const expectedStatus = 404;

  let server: App | undefined;

  afterEach(() => server?.exit());

  test('listen http (default addr)', async () => {
    server = await (await App.spawn()).waitForReady(3001);

    const resp = await client.get(`http://localhost:3001/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('listen http (specify addr)', async () => {
    const port = randomPort();
    server = await (
      await App.spawn(
        defaultConfig({
          deployment: { admin: { listen: `127.0.0.1:${port}` } },
        }),
      )
    ).waitForReady(port);

    const resp = await client.get(`http://localhost:${port}/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('empty listen', async () => {
    server = await (
      await App.spawn(
        defaultConfig({ deployment: { admin: { listen: undefined } } }),
      )
    ).waitForReady(3001);

    const resp = await client.get(`http://localhost:3001/`);
    expect(resp.status).toBe(expectedStatus);
  });

  test('invalid listen', async () => {
    server = undefined;

    await expect(
      App.spawn(
        defaultConfig({
          deployment: { admin: { listen: INVALID_LISTEN_ADDR } },
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
          { listen: '127.0.0.1:30000' },
        ),
      )
    ).waitForReady(3001);

    const resp = await client.get(`http://localhost:3001/`);
    expect(resp.status).toBe(expectedStatus);

    await expect(
      App.spawn(defaultConfig({ listen: '127.0.0.1:30001' }), 1000),
    ).rejects.toThrow(ERR_UNEXPECTED_EARLY_EXIT);
  });
});
