/**
 * E2E Frontend Streaming and Hook Lifecycle Tests (HI-03, HI-04)
 * Tests SSE stream trailing buffer flush and useSSEQuery unmount abort behavior.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { streamQuery } from "../frontend/src/lib/api.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function createMockStream(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      for (const chunk of chunks) {
        if (typeof chunk === "string") {
          controller.enqueue(encoder.encode(chunk));
        } else if (chunk instanceof Uint8Array) {
          controller.enqueue(chunk);
        }
      }
      controller.close();
    },
  });
}

async function testTrailingEventWithoutDoubleNewline() {
  // HI-03: Stream terminates with single newline or EOF without \n\n
  const payload = 'event: done\ndata: {"status": "completed", "total_tokens": 42}\n';
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(createMockStream([payload]), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const events = [];
    let completed = false;

    await new Promise((resolve, reject) => {
      streamQuery(
        "test question",
        {},
        (evt) => events.push(evt),
        () => {
          completed = true;
          resolve();
        },
        (err) => reject(err),
      );
    });

    assert.equal(completed, true, "Stream should complete");
    assert.equal(
      events.length,
      1,
      `Expected 1 event to be dispatched from trailing buffer, got ${events.length}`,
    );
    assert.equal(events[0].event, "done");
    assert.deepEqual(events[0].data, { status: "completed", total_tokens: 42 });
    console.log("  ✓ HI-03: Trailing SSE event without double newline parsed successfully");
  } finally {
    globalThis.fetch = origFetch;
  }
}

async function testUseSSEQueryUnmountAbortContract() {
  // HI-04: Verify useSSEQuery registers an unmount abort cleanup effect
  const hookFilePath = path.resolve(__dirname, "../frontend/src/hooks/useSSEQuery.ts");
  const content = fs.readFileSync(hookFilePath, "utf-8");

  const hasUseEffectImport = /import\s+{[^}]*useEffect[^}]*}\s+from\s+["']react["']/.test(content);
  assert.ok(
    hasUseEffectImport,
    "HI-04 Contract Violation: 'useEffect' must be imported in frontend/src/hooks/useSSEQuery.ts",
  );

  const hasAbortCleanup =
    /useEffect\s*\(\s*\(\)\s*=>\s*(?:\{\s*return\s*\(\)\s*=>\s*\{?\s*controllerRef\.current\?\.abort\(\)|(?:\(\)\s*=>\s*\{?\s*controllerRef\.current\?\.abort\(\)))/.test(
      content,
    ) ||
    /useEffect\s*\(\s*\(\)\s*=>\s*\{[\s\S]*controllerRef\.current\?\.abort\(\)[\s\S]*\}\s*,\s*\[\s*\]\s*\)/.test(
      content,
    );

  assert.ok(
    hasAbortCleanup,
    "HI-04 Contract Violation: useSSEQuery must register an unmount cleanup effect that aborts controllerRef.current",
  );
  console.log("  ✓ HI-04: useSSEQuery unmount abort handler verified");
}

async function run() {
  console.log("Running Frontend E2E Stream & Hook Lifecycle Tests...");
  let failed = 0;

  try {
    await testTrailingEventWithoutDoubleNewline();
  } catch (err) {
    console.error("  ✗ HI-03 FAILED:", err.message);
    failed++;
  }

  try {
    await testUseSSEQueryUnmountAbortContract();
  } catch (err) {
    console.error("  ✗ HI-04 FAILED:", err.message);
    failed++;
  }

  if (failed > 0) {
    process.exit(1);
  } else {
    console.log("All Frontend E2E Stream Tests Passed!");
  }
}

run();
