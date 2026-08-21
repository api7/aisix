# Glama deployment check

Files used by [Glama](https://glama.ai/mcp/servers/api7/aisix) to build and
health-check the AISIX MCP gateway in its sandbox (which wraps a **stdio**
MCP server and cannot run Docker):

- `setup.sh` — Glama build step: extracts the latest release `aisix` binary
  from the published GHCR image layers (`extract-aisix.sh`) and installs
  `mcp-remote`.
- `start.sh` — Glama start command: launches a tiny embedded MCP upstream
  (`echo-mcp.py`), starts the gateway with `config.yaml` + `resources.yaml`
  (anonymous access to the `echo` server only), and bridges Glama's stdio
  transport to the gateway's Streamable HTTP `/mcp` endpoint via `mcp-remote`.

This is a sandbox profile for the directory's automated checks — not a
deployment example. For real deployments, see the
[docs](https://docs.api7.ai/ai-gateway/).
