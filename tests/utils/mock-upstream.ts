export {
  buildOpenAiToolCallStreamEvents,
  OpenAiMockUpstream,
  startOpenAiMockUpstream,
  type OpenAiMockUpstreamOptions,
  type OpenAiMockStreamEvent,
  type RecordedRequest,
} from '../fixtures/mock-upstream.js';

export const buildOpenAiProviderModel = (model: string) => `openai/${model}`;

export const buildOpenAiProviderConfig = (
  apiBase: string,
  apiKey = 'upstream-key',
) => ({
  api_key: apiKey,
  api_base: apiBase,
});
