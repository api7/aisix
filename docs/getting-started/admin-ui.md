---
title: Admin UI
slug: /ai-gateway/getting-started/admin-ui
description: Build and use the AISIX Admin UI and Chat Playground.
---

AISIX includes a built-in **Admin UI** for managing models and API keys, and a **Chat Playground** for testing chat completions — all from your browser.

The UI is a React application that must be built before AISIX can serve it. Once built, it is embedded into the AISIX binary and served automatically.

## Prerequisites

- **Node.js**: LTS version.
- **pnpm**: Package manager. Install via `npm install -g pnpm` or see [pnpm.io](https://pnpm.io/installation).

## Build the UI

From the project root:

```bash
cd ui
pnpm install --frozen-lockfile
pnpm build
```

This compiles the React application into `ui/dist/`, which AISIX embeds at startup.

:::tip[Skip the UI Build]
If you only need the API and do not plan to use the Admin UI, you can skip the build and create a stub folder instead:

```bash
mkdir -p ui/dist
```

AISIX will start without the UI assets.
:::

## Start AISIX

After building the UI, start AISIX from the project root:

```bash
RUST_LOG=info cargo run
```

## Access the Admin UI

Open your browser and navigate to:

```
http://127.0.0.1:3001/ui
```

Log in with the `admin_key` from your `config.yaml` to access the dashboard.

## Features

### Model Management

Create, update, and delete models directly from the UI. Each model entry shows its name, provider, upstream model ID, and rate limit configuration.

### API Key Management

Manage API keys and their allowed model lists. You can create new keys, update permissions, and revoke access without restarting AISIX.

### Chat Playground

Test chat completions against any configured model. Select a model, type a message, and see the response in real time — including streaming output.

## UI Development

For local development with hot reload:

```bash
cd ui
pnpm dev
```

This starts the Vite dev server (default `http://localhost:5173`). The dev server proxies API requests to AISIX running on `127.0.0.1:3001`.
