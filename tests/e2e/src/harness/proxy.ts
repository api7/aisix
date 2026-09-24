import { harnessRequest } from "./http.js";

export interface ProxyResponse {
  status: number;
  body: unknown;
  /**
   * The gateway's own request id, echoed on every response. Lets a spec
   * wait for THIS request's access-log line — the FIFO barrier that says
   * everything the request wrote before it has been written too.
   */
  requestId: string;
}

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

  async listModels(): Promise<ProxyResponse> {
    return this.json("GET", "/v1/models");
  }

  async chat(body: unknown): Promise<ProxyResponse> {
    return this.json("POST", "/v1/chat/completions", body);
  }

  private async json(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<ProxyResponse> {
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
      requestId: res.headers["x-aisix-request-id"]?.toString() ?? "",
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
