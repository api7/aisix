/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// UI dev proxies the Admin API to avoid CORS.
// Build output is picked up by the `aisix-admin` crate via rust-embed.
export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: {
    outDir: "../crates/aisix-admin/ui-dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/admin": "http://127.0.0.1:3001",
      "/health": "http://127.0.0.1:3001",
      "/metrics": "http://127.0.0.1:3001",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    css: false,
  },
});
