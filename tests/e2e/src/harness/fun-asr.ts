import { randomUUID } from "node:crypto";
import { once, on } from "node:events";
import type { IncomingHttpHeaders } from "node:http";
import { WebSocket, WebSocketServer } from "ws";

export const funAsrPath = "/api-ws/v1/inference";

export function runTask(id: string, model: string) {
  return {
    header: { action: "run-task", task_id: id, streaming: "duplex" },
    payload: {
      task_group: "audio", task: "asr", function: "recognition", model,
      parameters: { format: "pcm", sample_rate: 16000 }, input: {},
    },
  };
}

export async function openFunAsr(base: string, key: string, headers: Record<string, string> = {}) {
  const ws = new WebSocket(base.replace("http", "ws") + funAsrPath, {
    headers: { ...headers, authorization: `Bearer ${key}` },
  });
  let requestId = "";
  ws.on("upgrade", res => { requestId = String(res.headers["x-aisix-request-id"]); });
  const closed = new AbortController();
  ws.once("close", () => closed.abort());
  const events = on(ws, "message", { signal: closed.signal });
  await once(ws, "open");
  return {
    ws, requestId,
    async frame() {
      const { value, done } = await events.next();
      if (done) throw new Error("WebSocket closed before the next event");
      return JSON.parse(String(value[0]));
    },
  };
}

export async function recognize(client: Awaited<ReturnType<typeof openFunAsr>>, model: string) {
  const id = randomUUID();
  client.ws.send(JSON.stringify(runTask(id, model)));
  const started = await client.frame();
  if (started.header.event !== "task-started") throw new Error(JSON.stringify(started));
  const audio = Buffer.from([0, 0xff, 0x80, 9]);
  client.ws.send(audio);
  client.ws.send(JSON.stringify({ header: { action: "finish-task", task_id: id, streaming: "duplex" }, payload: { input: {} } }));
  const frames = [started];
  do {
    frames.push(await client.frame());
  } while (frames.at(-1).header.event === "result-generated");
  return { id, frames, audio };
}

export async function startFunAsrUpstream() {
  const handshakes: IncomingHttpHeaders[] = [];
  const wss = new WebSocketServer({
    host: "127.0.0.1", port: 0,
    verifyClient(info, done) {
      handshakes.push(info.req.headers);
      if (info.req.url?.startsWith("/refuse")) {
        done(false, 403, JSON.stringify({ code: "AccessDenied", message: "workspace denied" }), {
          ...(info.req.url.startsWith("/refuse/") && { "x-request-id": "upstream-refused-id" }),
          "x-asr-region": "refused", "x-unlisted": "private",
        });
      } else {
        done(true);
      }
    },
  });
  wss.on("headers", headers => headers.push("x-asr-region: accepted", "x-unlisted: private"));
  const calls: { path: string; auth?: string; model: string; audio: Buffer[]; id: string }[] = [];
  wss.on("connection", (socket, req) => {
    const call = { path: req.url!, auth: req.headers.authorization, model: "", audio: [] as Buffer[], id: "" };
    calls.push(call);
    const send = (event: string, payload = {}, header = {}) =>
      socket.send(JSON.stringify({ header: { task_id: call.id, event, ...header }, payload }));
    socket.on("message", (data, binary) => {
      if (binary) {
        call.audio.push(Buffer.from(data as Buffer));
        if (call.model === "vendor-guard-output") {
          const frame = { header: { task_id: call.id, event: "result-generated" }, payload: {
            output: { sentence: { text: "测试词" } }, usage: { duration: 2 },
          } };
          socket.send(JSON.stringify(frame).replace("测试词", "\\u6d4b\\u8bd5\\u8bcd"));
        } else {
          send("result-generated", { output: { sentence: { text: "你好" } }, usage: { duration: 2 } });
        }
        return;
      }
      const { header, payload } = JSON.parse(String(data));
      if (header.action === "run-task") {
        call.id = header.task_id;
        call.model = payload.model;
        if (call.model === "vendor-guard-control") {
          const frame = { header: { task_id: call.id, event: "task-started", attributes: { note: "测试词" } }, payload: {} };
          socket.send(JSON.stringify(frame).replace("测试词", "\\u6d4b\\u8bd5\\u8bcd"));
        } else {
          send("task-started");
        }
      } else if (header.action === "finish-task") {
        send("result-generated", { output: { sentence: { text: "你好世界" } }, usage: { duration: 6 } });
        if (call.model === "vendor-failure") {
          send("task-failed", {}, { error_code: "BadRequest", error_message: "bad audio" });
        } else if (call.model === "vendor-unread-terminal") {
          // Stall the gateway's final write when the client has paused its receive socket.
          socket.send(" ".repeat(8 * 1024 * 1024) + JSON.stringify({ header: { task_id: call.id, event: "task-finished" }, payload: {} }));
        } else if (call.model === "vendor-drop") {
          socket.close();
        } else {
          send("task-finished");
        }
      }
    });
  });
  await once(wss, "listening");
  const addr = wss.address();
  if (!addr || typeof addr === "string") throw new Error("no listen port");
  return {
    base: `http://127.0.0.1:${addr.port}`, calls, handshakes,
    close: () => {
      for (const socket of wss.clients) socket.terminate();
      return new Promise<void>((resolve) => wss.close(() => resolve()));
    },
  };
}
