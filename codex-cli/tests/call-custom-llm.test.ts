import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { callCustomLLM } from "../src/utils/responses";
import { Readable } from "node:stream";
import { spawn } from "node:child_process";

vi.mock("node:child_process", () => {
  return { spawn: vi.fn() };
});

describe("callCustomLLM", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("yields parsed JSON lines from stdout", async () => {
    const fakeStdout = new Readable({ read() {} });
  (spawn as unknown as Mock).mockReturnValue({
      stdout: fakeStdout,
      stdin: { write: vi.fn(), end: vi.fn() },
    });

    const iterable = callCustomLLM({ foo: "bar" });

    fakeStdout.push('{"a":1}\n');
    fakeStdout.push('{"b":2}\n');
    fakeStdout.push(null);

    const results: Array<any> = [];
    for await (const item of iterable) {
      results.push(item);
    }

    expect(results).toEqual([{ a: 1 }, { b: 2 }]);
  });
});
