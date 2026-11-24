import { randomUUID } from "node:crypto";
import { once, on } from "node:events";
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

export async function openFunAsr(base: string, key: string) {
  const ws = new WebSocket(base.replace("http", "ws") + funAsrPath, {
    headers: { authorization: `Bearer ${key}` },
  });
  const closed = new AbortController();
  ws.once("close", () => closed.abort());
  const events = on(ws, "message", { signal: closed.signal });
  await once(ws, "open");
  return {
    ws,
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
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  const calls: { path: string; auth?: string; model: string; audio: Buffer[]; id: string }[] = [];
  wss.on("connection", (socket, req) => {
    const call = { path: req.url!, auth: req.headers.authorization, model: "", audio: [] as Buffer[], id: "" };
    calls.push(call);
    const send = (event: string, payload = {}, header = {}) =>
      socket.send(JSON.stringify({ header: { task_id: call.id, event, ...header }, payload }));
    socket.on("message", (data, binary) => {
      if (binary) {
        call.audio.push(Buffer.from(data as Buffer));
        send("result-generated", { output: { sentence: { text: "你好" } }, usage: { duration: 2 } });
        return;
      }
      const { header, payload } = JSON.parse(String(data));
      if (header.action === "run-task") {
        call.id = header.task_id;
        call.model = payload.model;
        send("task-started");
      } else if (header.action === "finish-task") {
        send("result-generated", { output: { sentence: { text: "你好世界" } }, usage: { duration: 6 } });
        if (call.model === "vendor-failure") {
          send("task-failed", {}, { error_code: "BadRequest", error_message: "bad audio" });
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
    base: `http://127.0.0.1:${addr.port}`, calls,
    close: () => {
      for (const socket of wss.clients) socket.terminate();
      return new Promise<void>((resolve) => wss.close(() => resolve()));
    },
  };
}
