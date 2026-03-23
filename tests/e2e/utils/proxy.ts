import type { AxiosRequestConfig } from 'axios';

import { client } from './http.js';

export const PROXY_BASE_URL = 'http://127.0.0.1:3000';

export const proxyAuthHeader = (key: string) => ({
  Authorization: `Bearer ${key}`,
});

export const proxyUrl = (path: string) => `${PROXY_BASE_URL}${path}`;

export const proxyGet = async (
  path: string,
  apiKey: string,
  config: AxiosRequestConfig = {},
) =>
  client.get(proxyUrl(path), {
    ...config,
    headers: {
      ...proxyAuthHeader(apiKey),
      ...(config.headers ?? {}),
    },
  });

export const proxyPost = async (
  path: string,
  body: unknown,
  apiKey: string,
  config: AxiosRequestConfig = {},
) =>
  client.post(proxyUrl(path), body, {
    ...config,
    headers: {
      ...proxyAuthHeader(apiKey),
      ...(config.headers ?? {}),
    },
  });

export const parseSseDataEvents = (sseBody: string) => {
  return sseBody
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith('data: '))
    .map((line) => line.slice('data: '.length));
};
