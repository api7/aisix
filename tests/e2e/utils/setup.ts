import { ChildProcess, spawn } from 'node:child_process';
import { once } from 'node:events';
import { unlink, writeFile } from 'node:fs/promises';

import deepmerge from '@fastify/deepmerge';
import { file } from 'tmp-promise';
import type { PartialDeep } from 'type-fest';

export interface AppConfig {
  deployment: {
    etcd: {
      host: string[];
      prefix: string;
      timeout: number;
    };
    admin?: {
      listen?: string;
      admin_key?: { key: string }[];
    };
  };
  listen?: string;
}

export const defaultConfig = (overrides?: PartialDeep<AppConfig>): AppConfig =>
  deepmerge()(
    {
      deployment: {
        etcd: {
          host: ['http://localhost:2379'],
          prefix: '/ai',
          timeout: 5000,
        },
      },
    },
    (overrides as AppConfig) ?? {},
  );

export const randomPort = () =>
  Math.floor(Math.random() * (65535 - 1024)) + 1024;

export class App {
  constructor(
    private readonly process: ChildProcess,
    private readonly configPath: string,
  ) {}

  static async spawn(config?: AppConfig): Promise<App> {
    const { path, cleanup } = await file({ postfix: '.json' });

    await writeFile(path, JSON.stringify(config ?? defaultConfig()));

    const appProcess = spawn('../../target/debug/ai-gateway', [
      '--config',
      path,
    ]);
    return new App(appProcess, path);
  }

  public async waitForReady(port?: number): Promise<App> {
    let times = 100;
    while (times-- > 0) {
      try {
        await fetch(`http://localhost:${port ?? 3000}`);
        return this;
      } catch (error) {}
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error('Server failed to start');
  }

  public async exit() {
    this.process.kill('SIGTERM');
    try {
      await Promise.race([
        once(this.process, 'exit'),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('timeout')), 3000),
        ),
      ]);
    } catch {
      this.process.kill('SIGKILL');
    }
    await unlink(this.configPath);
  }
}
