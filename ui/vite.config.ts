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
      "/aisix/admin": "http://127.0.0.1:3001",
      "/playground": "http://127.0.0.1:3001",
      "/openapi": "http://127.0.0.1:3001",
    },
  },
});
