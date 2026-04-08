export {
  MockUpstream,
  startMockUpstream as startOpenAiMockUpstream,
  type OpenAiMockUpstreamOptions,
  type RecordedRequest,
} from '../fixtures/mock-upstream.js';

export const buildOpenAiProviderModel = (model: string) => `openai/${model}`;

export const buildOpenAiProviderConfig = (
  baseUrl: string,
  apiKey = 'upstream-key',
) => ({
  api_key: apiKey,
  api_base: `${baseUrl}/v1`,
});