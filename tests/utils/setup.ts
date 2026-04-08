import { ChildProcess, spawn } from 'node:child_process';
import { once } from 'node:events';
import { unlink, writeFile } from 'node:fs/promises';
import { Agent as httpsAgent } from 'node:https';
import { text } from 'node:stream/consumers';

import deepmerge from '@fastify/deepmerge';
import { file } from 'tmp-promise';
import type { PartialDeep } from 'type-fest';

import { client } from './http.js';

export interface AppConfig {
  deployment: {
    etcd: {
      host: string[];
      prefix: string;
      timeout: number;
    };
    admin?: {
      admin_key?: { key: string }[];
    };
  };
  server: {
    proxy: {
      listen?: string;
      tls?: {
        enabled?: boolean;
        cert_file?: string;
        key_file?: string;
      };
    };
    admin: {
      listen?: string;
      tls?: {
        enabled?: boolean;
        cert_file?: string;
        key_file?: string;
      };
    };
  };
}

export const defaultConfig = (overrides?: PartialDeep<AppConfig>): AppConfig =>
  deepmerge()(
    {
      deployment: {
        etcd: {
          host: ['http://127.0.0.1:2379'],
          prefix: '/ai',
          timeout: 5000,
        },
      },
    },
    (overrides as AppConfig) ?? {},
  );

export const tlsSkipVerify = new httpsAgent({
  rejectUnauthorized: false,
});

export const randomPort = () =>
  Math.floor(Math.random() * (65535 - 1024)) + 1024;

export const ERR_UNEXPECTED_EXIT = 'Process exited unexpectedly';
export const ERR_UNEXPECTED_EARLY_EXIT = 'Process exited early with code';

export enum AppState {
  RUNNING,
  EXITED,
}

export class App {
  private processState = AppState.RUNNING;

  constructor(
    private readonly process: ChildProcess,
    private readonly configPath: string,
  ) {
    once(this.process, 'exit').then(() => {
      this.processState = AppState.EXITED;
    });
  }

  static async spawn(config?: AppConfig, stableMs = 0): Promise<App> {
    const { path, cleanup } = await file({ postfix: '.json' });

    await writeFile(path, JSON.stringify(config ?? defaultConfig()));

    const appProcess = spawn('../target/debug/aisix', ['--config', path]);

    return new Promise<App>((resolve, reject) => {
      let exited = false;

      appProcess.on('exit', async (code) => {
        exited = true;
        cleanup();

        const stdout = await text(appProcess.stdout!);
        const stderr = await text(appProcess.stderr!);
        reject(
          new Error(
            `Process exited early with code ${code}.\nStdout: ${stdout}.\nStderr: ${stderr}`,
          ),
        );
      });
      appProcess.on('error', reject);

      setTimeout(() => {
        if (!exited) {
          appProcess.removeAllListeners('exit');
          resolve(new App(appProcess, path));
        }
      }, stableMs);
    });
  }

  public async waitForReady(portOrURL?: number | string): Promise<App> {
    let times = 100;
    while (times-- > 0) {
      // If the process exits while waiting for ready, this is considered an unexpected exit.
      if (this.processState === AppState.EXITED)
        throw new Error(ERR_UNEXPECTED_EXIT);

      try {
        await client.get(
          portOrURL
            ? typeof portOrURL == 'number'
              ? `http://127.0.0.1:${portOrURL}`
              : portOrURL
            : `http://127.0.0.1:3000/`,
          { httpsAgent: tlsSkipVerify },
        );
        return this;
      } catch (_err) {
        //
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error('Server failed to start');
  }

  public pid() {
    return this.process.pid;
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
