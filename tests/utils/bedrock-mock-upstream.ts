export {
  BedrockMockUpstream,
  startBedrockMockUpstream,
  type BedrockMockUpstreamOptions,
  type BedrockStreamEvent,
  type RecordedRequest,
} from '../fixtures/bedrock-mock-upstream.js';

interface BedrockProviderConfigOverrides {
  region?: string;
  access_key_id?: string;
  secret_access_key?: string;
  session_token?: string;
}

export const buildBedrockProviderModel = (model: string) => model;

export const buildBedrockProviderConfig = (
  endpoint: string,
  overrides: BedrockProviderConfigOverrides = {},
) => ({
  region: overrides.region ?? 'us-east-1',
  access_key_id: overrides.access_key_id ?? 'AKIA123',
  secret_access_key: overrides.secret_access_key ?? 'secret',
  session_token: overrides.session_token ?? 'session-token',
  endpoint,
});
