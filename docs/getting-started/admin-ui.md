---
title: Admin UI
slug: /aisix/getting-started/admin-ui
description: Use the AISIX Admin UI and Chat Playground to manage models, API keys, and test completions.
---

AISIX includes a built-in **Admin UI** for managing models and API keys, and a **Chat Playground** for testing chat completions — all from your browser.

## Access the Admin UI

If you started AISIX using the [Quick Start](./quick-start.md) Docker setup, the UI is already running. Open your browser and navigate to:

```
http://127.0.0.1:3001/ui
```

Log in with the Admin Key printed by the quickstart script.

## Features

### Model Management

Create, update, and delete models directly from the UI. Each model entry shows its name, provider, upstream model ID, and rate limit configuration.

### API Key Management

Manage API keys and their allowed model lists. You can create new keys, update permissions, and revoke access without restarting AISIX.

### Chat Playground

Test chat completions against any configured model. Select a model, type a message, and see the response in real time — including streaming output.

## UI Development

If you are running AISIX from source and want to develop the UI with hot reload:

**Prerequisites:**
- **Node.js**: LTS version.
- **pnpm**: Install via `npm install -g pnpm` or see [pnpm.io](https://pnpm.io/installation).

**Build the UI** (required before running from source):

```bash
cd ui
pnpm install --frozen-lockfile
pnpm build
```

**Start the dev server** with hot reload:

```bash
cd ui
pnpm dev
```

This starts the Vite dev server at `http://localhost:5173`. API requests are proxied to AISIX on `127.0.0.1:3001`.
