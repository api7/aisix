import { harnessRequest } from "./http.js";

/**
 * Thin typed wrapper over the proxy surface. Tests that want full SDK
 * compatibility can use the `openai` npm package directly with
 * `{ baseURL: app.proxyUrl + "/v1" }` — this client is for the cases
 * where we want to inspect raw status codes, headers, or non-OpenAI
 * endpoints (e.g. `/v1/messages`, `/passthrough/...`).
 */
export class ProxyClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
  ) {}

  async listModels(): Promise<{ status: number; body: unknown }> {
    return this.json("GET", "/v1/models");
  }

  async chat(body: unknown): Promise<{ status: number; body: unknown }> {
    return this.json("POST", "/v1/chat/completions", body);
  }

  private async json(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<{ status: number; body: unknown }> {
    const res = await harnessRequest(`${this.baseUrl}${path}`, {
      method,
      headers: {
        authorization: `Bearer ${this.apiKey}`,
        "content-type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.body.text();
    return {
      status: res.statusCode,
      body: text ? safeParse(text) : null,
    };
  }
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
