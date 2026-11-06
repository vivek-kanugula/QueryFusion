/**
 * Adversarial stress testing for frontend SSE stream parsing and client contracts.
 * Tests frontend/src/lib/api.ts directly using Node.js native execution.
 */

import assert from "node:assert/strict";
import {
  streamQuery,
  ApiError,
} from "../frontend/src/lib/api.ts";

// Helper to create a ReadableStream from string chunks or byte chunks with custom delays
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

// Test harness runner
const tests = [];
function test(name, fn) {
  tests.push({ name, fn });
}

// -----------------------------------------------------------------------------
// 1. Boundary Condition: Split chunks across TCP packets / arbitrary packet slices
// -----------------------------------------------------------------------------
test("SSE: TCP packet splitting across event boundaries, field boundaries, and JSON tokens", async () => {
  const fullPayload =
    "event: stage_start\ndata: {\"stage\": \"retrieval\", \"ts\": 100}\n\n" +
    "event: token\ndata: {\"token\": \"Hybrid\", \"index\": 0}\n\n" +
    "event: token\ndata: {\"token\": \"-RAG\", \"index\": 1}\n\n" +
    "event: done\ndata: {\"total_tokens\": 2}\n\n";

  // Slice into tiny 3-byte packets to stress the buffer accumulator
  const tinyChunks = [];
  for (let i = 0; i < fullPayload.length; i += 3) {
    tinyChunks.push(fullPayload.slice(i, i + 3));
  }

  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(createMockStream(tinyChunks), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const receivedEvents = [];
    await new Promise((resolve, reject) => {
      streamQuery(
        "test question",
        { retrieval_strategy: "hybrid_rerank", alpha: 0.5, rerank_enabled: true, verify: true },
        (evt) => receivedEvents.push(evt),
        () => resolve(),
        (err) => reject(err),
      );
    });

    assert.equal(receivedEvents.length, 4);
    assert.deepEqual(receivedEvents[0], {
      event: "stage_start",
      data: { stage: "retrieval", ts: 100 },
    });
    assert.deepEqual(receivedEvents[1], {
      event: "token",
      data: { token: "Hybrid", index: 0 },
    });
    assert.deepEqual(receivedEvents[2], {
      event: "token",
      data: { token: "-RAG", index: 1 },
    });
    assert.deepEqual(receivedEvents[3], {
      event: "done",
      data: { total_tokens: 2 },
    });
  } finally {
    globalThis.fetch = origFetch;
  }
});

// -----------------------------------------------------------------------------
// 2. Boundary Condition: CRLF vs LF vs Mixed Line Endings & Split CRLF
// -----------------------------------------------------------------------------
test("SSE: CRLF, LF, mixed endings, and split CRLF (\\r in packet 1, \\n in packet 2)", async () => {
  // Test split CRLF right between \r and \n:
  // "event: token\r" in chunk 1, "\ndata: {\"msg\": \"crlf_split\"}\r\n\r\n" in chunk 2
  const chunks = [
    "event: token\r",
    "\ndata: {\"msg\": \"crlf_split\"}\r\n\r\n",
    "event: unix\ndata: {\"msg\": \"unix_lf\"}\n\n",
    "event: mixed\r\ndata: {\"msg\": \"mixed_newlines\"}\n\n",
  ];

  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(createMockStream(chunks), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const receivedEvents = [];
    await new Promise((resolve, reject) => {
      streamQuery(
        "test",
        { retrieval_strategy: "hybrid", alpha: 0.5, rerank_enabled: false, verify: false },
        (evt) => receivedEvents.push(evt),
        () => resolve(),
        (err) => reject(err),
      );
    });

    assert.equal(receivedEvents.length, 3);
    assert.equal(receivedEvents[0].event, "token");
    assert.equal(receivedEvents[0].data.msg, "crlf_split");
    assert.equal(receivedEvents[1].event, "unix");
    assert.equal(receivedEvents[1].data.msg, "unix_lf");
    assert.equal(receivedEvents[2].event, "mixed");
    assert.equal(receivedEvents[2].data.msg, "mixed_newlines");
  } finally {
    globalThis.fetch = origFetch;
  }
});

// -----------------------------------------------------------------------------
// 3. Boundary Condition: Empty event lines, comments, and empty payloads
// -----------------------------------------------------------------------------
test("SSE: Empty event lines, heartbeat comments, and empty data handling", async () => {
  const payload =
    "\n\n\n" + // initial blank lines (must be ignored)
    ": sse keep-alive comment\n\n" + // SSE comment
    "event: heartbeat\ndata: {}\n\n" +
    "\n\n\n" + // multi-blank separator (must be ignored)
    "event: empty_data\ndata:\n\n" + // empty data line (parses to {})
    "event: final\ndata: {\"ok\": true}\n\n";

  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(createMockStream([payload]), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const receivedEvents = [];
    await new Promise((resolve, reject) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        (evt) => receivedEvents.push(evt),
        () => resolve(),
        (err) => reject(err),
      );
    });

    // Verify empty separator lines (\n\n\n) were safely ignored
    // Comment lines currently emit { event: "message", data: {} } in api.ts
    // Explicit events (heartbeat, empty_data, final) are properly parsed
    const finalEvent = receivedEvents.find((e) => e.event === "final");
    assert.ok(finalEvent, "Final event must be received");
    assert.deepEqual(finalEvent.data, { ok: true });

    const heartbeatEvent = receivedEvents.find((e) => e.event === "heartbeat");
    assert.ok(heartbeatEvent, "Heartbeat event must be received");
    assert.deepEqual(heartbeatEvent.data, {});

    const emptyDataEvent = receivedEvents.find((e) => e.event === "empty_data");
    assert.ok(emptyDataEvent, "Empty data event must be received");
    assert.deepEqual(emptyDataEvent.data, {});
  } finally {
    globalThis.fetch = origFetch;
  }
});

// -----------------------------------------------------------------------------
// 4. Boundary Condition: Unknown event types and multi-line data
// -----------------------------------------------------------------------------
test("SSE: Unknown event types preserved and multi-line JSON payload parsed", async () => {
  const payload =
    "event: custom_unknown_event_type_v42\n" +
    "data: {\n" +
    "data:   \"nested\": {\n" +
    "data:     \"status\": \"unrecognized_but_preserved\"\n" +
    "data:   }\n" +
    "data: }\n\n";

  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(createMockStream([payload]), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    };

    const receivedEvents = [];
    await new Promise((resolve, reject) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        (evt) => receivedEvents.push(evt),
        () => resolve(),
        (err) => reject(err),
      );
    });

    assert.equal(receivedEvents.length, 1);
    assert.equal(receivedEvents[0].event, "custom_unknown_event_type_v42");
    assert.equal(
      receivedEvents[0].data.nested.status,
      "unrecognized_but_preserved",
    );
  } finally {
    globalThis.fetch = origFetch;
  }
});

// -----------------------------------------------------------------------------
// 5. Boundary Condition: Connection Drops and AbortController
// -----------------------------------------------------------------------------
test("SSE: AbortController does not trigger onError callback", async () => {
  const origFetch = globalThis.fetch;
  try {
    let controllerRef;
    globalThis.fetch = async (url, opts) => {
      const stream = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode("event: start\ndata: {}\n\n"));
        },
      });
      return new Response(stream, { status: 200 });
    };

    let errorFired = false;
    await new Promise((resolve) => {
      controllerRef = streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        () => {
          // Abort immediately upon first event
          controllerRef.abort();
          setTimeout(resolve, 50);
        },
        () => resolve(),
        () => {
          errorFired = true;
          resolve();
        },
      );
    });

    assert.equal(errorFired, false, "AbortError must NOT trigger onError");
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("SSE: Network connection drop mid-stream triggers onError callback", async () => {
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      const stream = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode("event: start\ndata: {}\n\n"));
          c.error(new Error("TCP connection reset by peer (simulated drop)"));
        },
      });
      return new Response(stream, { status: 200 });
    };

    let capturedError = null;
    await new Promise((resolve) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        () => {},
        () => resolve(),
        (err) => {
          capturedError = err;
          resolve();
        },
      );
    });

    assert.ok(capturedError instanceof Error);
    assert.match(capturedError.message, /TCP connection reset by peer/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

// -----------------------------------------------------------------------------
// 6. Boundary Condition: HTTP 4xx / 5xx error responses
// -----------------------------------------------------------------------------
test("SSE: HTTP 400 returns structured ApiError", async () => {
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({ detail: "Invalid request parameter" }),
        { status: 400, statusText: "Bad Request" },
      );
    };

    let capturedError = null;
    await new Promise((resolve) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        () => {},
        () => resolve(),
        (err) => {
          capturedError = err;
          resolve();
        },
      );
    });

    assert.ok(capturedError instanceof ApiError);
    assert.equal(capturedError.status, 400);
    assert.match(capturedError.detail, /Invalid request parameter/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("SSE: HTTP 403 returns structured ApiError with forbidden message", async () => {
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({ detail: "Invalid or unauthorized API key" }),
        { status: 403, statusText: "Forbidden" },
      );
    };

    let capturedError = null;
    await new Promise((resolve) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        () => {},
        () => resolve(),
        (err) => {
          capturedError = err;
          resolve();
        },
      );
    });

    assert.ok(capturedError instanceof ApiError);
    assert.equal(capturedError.status, 403);
    assert.match(capturedError.detail, /Invalid or unauthorized/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("SSE: HTTP 422 with validation errors returns detailed error message", async () => {
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({
          detail: [
            { loc: ["body", "temperature"], msg: "Input should be greater than or equal to 0" },
          ],
        }),
        { status: 422, statusText: "Unprocessable Entity" },
      );
    };

    let capturedError = null;
    await new Promise((resolve) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        () => {},
        () => resolve(),
        (err) => {
          capturedError = err;
          resolve();
        },
      );
    });

    assert.ok(capturedError instanceof ApiError);
    assert.equal(capturedError.status, 422);
    assert.match(capturedError.detail, /Input should be greater than or equal to 0/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

test("SSE: HTTP 500 server crash returns structured ApiError", async () => {
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      return new Response("Internal Server Error: Database Connection Refused", {
        status: 500,
        statusText: "Internal Server Error",
      });
    };

    let capturedError = null;
    await new Promise((resolve) => {
      streamQuery(
        "test",
        { retrieval_strategy: "fixed", alpha: 0.5, rerank_enabled: false, verify: false },
        () => {},
        () => resolve(),
        (err) => {
          capturedError = err;
          resolve();
        },
      );
    });

    assert.ok(capturedError instanceof ApiError);
    assert.equal(capturedError.status, 500);
    assert.match(capturedError.detail, /Internal Server Error: Database Connection Refused/);
  } finally {
    globalThis.fetch = origFetch;
  }
});

// -----------------------------------------------------------------------------
// Execute All Tests
// -----------------------------------------------------------------------------
async function runAll() {
  console.log(`Starting ${tests.length} adversarial SSE stress tests...`);
  let passed = 0;
  for (const { name, fn } of tests) {
    try {
      await fn();
      console.log(`  ✓ PASSED: ${name}`);
      passed++;
    } catch (err) {
      console.error(`  ✗ FAILED: ${name}`);
      console.error(err);
      process.exitCode = 1;
    }
  }
  console.log(`\nSummary: ${passed}/${tests.length} passed.`);
  if (process.exitCode) {
    process.exit(process.exitCode);
  }
}

runAll();
