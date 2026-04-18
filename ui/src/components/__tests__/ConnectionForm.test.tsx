import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConnectionForm } from "../ConnectionForm";

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ConnectionForm", () => {
  it("renders the supplied initial values", () => {
    render(
      <ConnectionForm
        initial={{ endpoint: "http://example.test:9999", adminKey: "k1" }}
        onSaved={() => {}}
      />,
    );
    expect(screen.getByDisplayValue("http://example.test:9999")).toBeInTheDocument();
    expect(screen.getByDisplayValue("k1")).toBeInTheDocument();
  });

  it("blocks submit with empty admin key and shows a validation error", () => {
    const onSaved = vi.fn();
    render(
      <ConnectionForm
        initial={{ endpoint: "http://example.test", adminKey: "" }}
        onSaved={onSaved}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /save/i }));
    expect(onSaved).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/admin key/i);
  });

  it("persists trimmed values to localStorage and notifies parent on save", () => {
    const onSaved = vi.fn();
    render(
      <ConnectionForm
        initial={{ endpoint: "  http://example.test  ", adminKey: " k1 " }}
        onSaved={onSaved}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    const stored = window.localStorage.getItem("aisix.connection");
    expect(stored).not.toBeNull();
    const parsed = JSON.parse(stored as string);
    expect(parsed.endpoint).toBe("http://example.test");
    expect(parsed.adminKey).toBe("k1");

    expect(onSaved).toHaveBeenCalledWith({
      endpoint: "http://example.test",
      adminKey: "k1",
    });
  });

  it("blocks submit when endpoint is whitespace-only", () => {
    const onSaved = vi.fn();
    render(
      <ConnectionForm
        initial={{ endpoint: "   ", adminKey: "k1" }}
        onSaved={onSaved}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /save/i }));
    expect(onSaved).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/endpoint/i);
  });
});
