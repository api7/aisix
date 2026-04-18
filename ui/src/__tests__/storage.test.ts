import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearConnection,
  DEFAULT_CONNECTION,
  isConfigured,
  loadConnection,
  saveConnection,
} from "../storage";

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  window.localStorage.clear();
});

describe("storage", () => {
  it("returns the default connection when nothing is stored", () => {
    expect(loadConnection()).toEqual(DEFAULT_CONNECTION);
  });

  it("round-trips a connection through save/load", () => {
    saveConnection({ endpoint: "http://x", adminKey: "k" });
    expect(loadConnection()).toEqual({ endpoint: "http://x", adminKey: "k" });
  });

  it("falls back to defaults when stored JSON is corrupt", () => {
    window.localStorage.setItem("aisix.connection", "{not json");
    expect(loadConnection()).toEqual(DEFAULT_CONNECTION);
  });

  it("clearConnection removes the entry", () => {
    saveConnection({ endpoint: "http://x", adminKey: "k" });
    clearConnection();
    expect(window.localStorage.getItem("aisix.connection")).toBeNull();
  });

  it("isConfigured requires both endpoint and adminKey to be non-empty", () => {
    expect(isConfigured({ endpoint: "", adminKey: "" })).toBe(false);
    expect(isConfigured({ endpoint: "http://x", adminKey: "" })).toBe(false);
    expect(isConfigured({ endpoint: "", adminKey: "k" })).toBe(false);
    expect(isConfigured({ endpoint: "http://x", adminKey: "k" })).toBe(true);
  });
});
