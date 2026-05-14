import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const ETCDCTL = process.env.ETCDCTL_PATH ?? 'etcdctl';
const ETCD_ENDPOINTS = process.env.ETCD_ENDPOINTS ?? 'http://127.0.0.1:2379';

export const etcdPutJson = async (
  prefix: string,
  path: string,
  value: unknown,
) => {
  await execFileAsync(
    ETCDCTL,
    [
      '--endpoints',
      ETCD_ENDPOINTS,
      '--dial-timeout=5s',
      '--command-timeout=5s',
      'put',
      `${prefix}${path}`,
      JSON.stringify(value),
    ],
    { timeout: 10_000 },
  );
};
