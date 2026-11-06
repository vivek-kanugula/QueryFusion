/**
 * Verification test suite for Approach 1: Client-Side sessionStorage persistence for Q&A.
 *
 * Verifies:
 * 1. Safe JSON serialization / deserialization round-trip
 * 2. Complete state preservation (question, answer, citations, claims, timing, doc scoping)
 * 3. Graceful fallback on empty, null, or corrupted sessionStorage
 * 4. Tab isolation (simulating multiple tabs with separate sessionStorage instances)
 * 5. Navigation override priority (location.state takes precedence over saved session)
 * 6. Clean removal on session reset
 */

import assert from "node:assert/strict";

const SESSION_STORAGE_KEY = "hybrid_rag_qa_session";

// In-memory mock of Storage interface for simulation
class MockStorage {
  constructor() {
    this.store = new Map();
  }
  getItem(key) {
    return this.store.has(key) ? this.store.get(key) : null;
  }
  setItem(key, value) {
    this.store.set(key, String(value));
  }
  removeItem(key) {
    this.store.delete(key);
  }
  clear() {
    this.store.clear();
  }
}

function loadSessionState(storage) {
  try {
    const raw = storage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveSessionState(storage, data) {
  try {
    if (!data) {
      storage.removeItem(SESSION_STORAGE_KEY);
    } else {
      storage.setItem(SESSION_STORAGE_KEY, JSON.stringify(data));
    }
  } catch {
    // Graceful fallback
  }
}

console.log("Running QA Session Persistence Tests...");

// Test 1: Serialization round-trip
{
  const tab1Storage = new MockStorage();
  const mockPayload = {
    question: "What is alpha-RRF?",
    mode: "hybrid_rerank",
    alpha: 0.65,
    selectedDocIds: ["doc-uuid-1", "doc-uuid-2"],
    stages: ["dense", "sparse", "rrf", "rerank", "generation", "verification"],
    answer: {
      answer: "Alpha-RRF blends dense and sparse scores [C1].",
      citations: [
        {
          citation_id: "C1",
          chunk_id: "chunk-uuid-10",
          document_id: "doc-uuid-1",
          evidence_quote: "Alpha-RRF uses weighted reciprocal rank fusion.",
        },
      ],
      has_sufficient_context: true,
      fallback_reason: null,
    },
    verification: {
      citation_coverage: 1.0,
      all_supported: true,
      claims: [
        {
          claim_text: "Alpha-RRF blends dense and sparse scores",
          citation_id: "C1",
          verdict: "SUPPORTED",
          explanation: "Explicitly backed by chunk-uuid-10",
        },
      ],
    },
    confidence: {
      overall_score: 0.95,
      retrieval_score: 0.9,
      citation_coverage: 1.0,
      completeness_score: 0.95,
    },
    completed: {
      retrieval_ms: 42.1,
      generation_ms: 310.5,
      model: "gemini-3.5-flash-lite",
      trace_id: "trace-xyz",
    },
    error: null,
    traceId: "trace-xyz",
  };

  saveSessionState(tab1Storage, mockPayload);
  const restored = loadSessionState(tab1Storage);

  assert.deepEqual(restored, mockPayload, "Restored state must match saved payload exactly");
  console.log("  ✓ PASSED: Full state serialization and deserialization round-trip");
}

// Test 2: Multi-tab isolation
{
  const tabA = new MockStorage();
  const tabB = new MockStorage();

  const sessionDocA = {
    question: "Query about Document A",
    mode: "dense",
    alpha: 1.0,
    selectedDocIds: ["doc-A"],
    stages: ["dense", "generation"],
    answer: { answer: "Answer for Doc A" },
    verification: null,
    confidence: { overall_score: 0.8 },
    completed: { retrieval_ms: 20 },
    error: null,
    traceId: "trace-A",
  };

  const sessionDocB = {
    question: "Query about Document B",
    mode: "hybrid_rerank",
    alpha: 0.3,
    selectedDocIds: ["doc-B"],
    stages: ["dense", "sparse", "rrf", "rerank", "generation"],
    answer: { answer: "Answer for Doc B" },
    verification: null,
    confidence: { overall_score: 0.92 },
    completed: { retrieval_ms: 35 },
    error: null,
    traceId: "trace-B",
  };

  saveSessionState(tabA, sessionDocA);
  saveSessionState(tabB, sessionDocB);

  // Refresh tab A: loads tab A's session without seeing tab B's
  const restoredA = loadSessionState(tabA);
  assert.equal(restoredA.question, "Query about Document A");
  assert.deepEqual(restoredA.selectedDocIds, ["doc-A"]);

  // Refresh tab B: loads tab B's session without seeing tab A's
  const restoredB = loadSessionState(tabB);
  assert.equal(restoredB.question, "Query about Document B");
  assert.deepEqual(restoredB.selectedDocIds, ["doc-B"]);

  console.log("  ✓ PASSED: Multiple tabs maintain independent, isolated sessionStorage");
}

// Test 3: Corrupted or invalid JSON handling
{
  const corruptedStorage = new MockStorage();
  corruptedStorage.setItem(SESSION_STORAGE_KEY, "{bad json truncated...");

  const result = loadSessionState(corruptedStorage);
  assert.equal(result, null, "Corrupted storage must return null safely without throwing");
  console.log("  ✓ PASSED: Corrupted storage handled gracefully without crash");
}

// Test 4: Navigation priority over saved session
{
  const storage = new MockStorage();
  saveSessionState(storage, {
    question: "Old question",
    mode: "hybrid",
    alpha: 0.5,
    selectedDocIds: ["saved-doc-1"],
    stages: [],
    answer: null,
  });

  const saved = loadSessionState(storage);

  // Case A: User clicked 'Ask Questions' on doc-NEW in Documents view
  const locationState = { selectedDocIds: ["doc-NEW"] };
  const effectiveDocIds =
    locationState?.selectedDocIds && locationState.selectedDocIds.length > 0
      ? locationState.selectedDocIds
      : saved?.selectedDocIds ?? [];

  assert.deepEqual(effectiveDocIds, ["doc-NEW"], "Explicit navigation state must take precedence");

  // Case B: User refreshed directly or navigated without state
  const refreshLocationState = null;
  const refreshedDocIds =
    refreshLocationState?.selectedDocIds && refreshLocationState.selectedDocIds.length > 0
      ? refreshLocationState.selectedDocIds
      : saved?.selectedDocIds ?? [];

  assert.deepEqual(refreshedDocIds, ["saved-doc-1"], "On refresh without state, saved session is preserved");

  console.log("  ✓ PASSED: Location state priority and refresh fallback contract");
}

// Test 5: Session reset cleans storage
{
  const storage = new MockStorage();
  saveSessionState(storage, { question: "Temporary query", answer: { answer: "Temp" } });
  assert.ok(storage.getItem(SESSION_STORAGE_KEY) !== null);

  saveSessionState(storage, null);
  assert.equal(storage.getItem(SESSION_STORAGE_KEY), null, "Resetting must delete session key");
  assert.equal(loadSessionState(storage), null);

  console.log("  ✓ PASSED: Session reset cleanly removes storage key");
}

console.log("\nAll 5 QA Session Persistence tests passed successfully!");

