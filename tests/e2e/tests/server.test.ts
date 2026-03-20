import { client } from '../utils/http.js';
import { App, AppConfig, defaultConfig, randomPort } from '../utils/setup.js';

describe('proxy server', () => {
  let server: App;

  afterEach(() => server?.exit());

  test('listen http (default addr)', async () => {
    server = await (await App.spawn()).waitForReady();

    const resp = await client.get(`http://localhost:3000/`);
    expect(resp.status).toBe(401);
  });

  test('listen http (specify addr)', async () => {
    const port = randomPort();
    server = await (
      await App.spawn(defaultConfig({ listen: `127.0.0.1:${port}` }))
    ).waitForReady(port);

    const resp = await client.get(`http://localhost:${port}/`);
    expect(resp.status).toBe(401);
  });
});

describe('admin server', () => {
  let server: App;

  afterEach(() => server?.exit());

  test('listen http (default addr)', async () => {
    server = await (await App.spawn()).waitForReady(3001);

    const resp = await client.get(`http://localhost:3001/`);
    expect(resp.status).toBe(404);
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
    expect(resp.status).toBe(404);
  });
});
