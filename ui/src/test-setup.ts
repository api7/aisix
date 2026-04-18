import "@testing-library/jest-dom/vitest";

// Vitest's jsdom environment doesn't provide a usable `confirm`; stub
// it so component tests that exercise delete actions can run without
// pulling in a separate dialog mock.
if (typeof window !== "undefined") {
  window.confirm = () => true;
}
