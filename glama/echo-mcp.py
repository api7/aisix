#!/usr/bin/env python3
"""Minimal streamable-HTTP MCP upstream: one `echo` tool. For the Glama check image only."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

TOOLS = [{
    "name": "echo",
    "description": "Echo the input text back.",
    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        rid, method = req.get("id"), req.get("method", "")
        if method == "initialize":
            self._json(200, {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": req.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo-upstream", "version": "1.0.0"}}})
        elif method == "notifications/initialized":
            self.send_response(202); self.send_header("Content-Length", "0"); self.end_headers()
        elif method == "tools/list":
            self._json(200, {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            text = req.get("params", {}).get("arguments", {}).get("text", "")
            self._json(200, {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}})
        else:
            self._json(200, {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}})

HTTPServer(("127.0.0.1", 13100), H).serve_forever()
