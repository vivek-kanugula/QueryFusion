/**
 * Stress test suite for citation resolution and parsing in frontend/src/views/QA.tsx.
 * 
 * Verifies:
 * 1. Empty citation array & null/undefined citations
 * 2. Citation IDs exceeding list length (e.g. [999], [0], [1e15])
 * 3. Negative numbers, non-numeric strings, weird/hostile characters
 * 4. Non-standard formats [C10], [P-1], [^2], [doc_42], [chunk_abc]
 * 5. Multiple citations on the same line, adjacent citations, nested brackets, code blocks
 * 6. Crash immunity & unhandled exception resilience
 */

import assert from "node:assert";

// Extract exact resolveCitation logic from QA.tsx
const resolveCitation = (rawToken, citations) => {
  if (!citations || citations.length === 0) return undefined;

  const cleanToken = rawToken.replace(/^\[|\]$/g, "").trim();
  const lowerClean = cleanToken.toLowerCase();

  // 1. Exact match on citation_id (raw or bracket-stripped)
  const exact = citations.find((c) => {
    if (!c || !c.citation_id) return false;
    const cid = c.citation_id.trim();
    const cleanCid = cid.replace(/^\[|\]$/g, "").trim();
    return (
      cid === rawToken ||
      cleanCid === cleanToken ||
      cid.toLowerCase() === rawToken.toLowerCase() ||
      cleanCid.toLowerCase() === lowerClean
    );
  });
  if (exact) {
    return {
      citation_id: exact.citation_id || rawToken,
      chunk_id: exact.chunk_id || "",
      document_id: exact.document_id,
      evidence_quote: exact.evidence_quote || "",
    };
  }

  // 2. Direct match on chunk_id or document_id
  const byId = citations.find(
    (c) =>
      c &&
      ((c.chunk_id && (c.chunk_id === rawToken || c.chunk_id === cleanToken)) ||
      (c.document_id && (c.document_id === rawToken || c.document_id === cleanToken))),
  );
  if (byId) {
    return {
      citation_id: byId.citation_id || rawToken,
      chunk_id: byId.chunk_id || "",
      document_id: byId.document_id,
      evidence_quote: byId.evidence_quote || "",
    };
  }

  // 3. Normalized numeric comparison (e.g. "[C1]", "[1]", "[^1]")
  const numMatch = cleanToken.match(/^(?:c|\^|doc-)?(\d+)$/i);
  if (numMatch) {
    const num = parseInt(numMatch[1], 10);
    const byNum = citations.find((c) => {
      if (!c || !c.citation_id) return false;
      const cidClean = c.citation_id.replace(/^\[|\]$/g, "").trim();
      const m = cidClean.match(/^(?:c|\^|doc-)?(\d+)$/i);
      return m && parseInt(m[1], 10) === num;
    });
    if (byNum) {
      return {
        citation_id: byNum.citation_id || rawToken,
        chunk_id: byNum.chunk_id || "",
        document_id: byNum.document_id,
        evidence_quote: byNum.evidence_quote || "",
      };
    }

    // 4. Safe 1-based index fallback
    const idx = num - 1;
    if (idx >= 0 && idx < citations.length && citations[idx]) {
      const fb = citations[idx];
      return {
        citation_id: fb.citation_id || rawToken,
        chunk_id: fb.chunk_id || "",
        document_id: fb.document_id,
        evidence_quote: fb.evidence_quote || "",
      };
    }
  }

  return undefined;
};

// Exact regex replacement from renderAnswer
const preprocessAnswer = (text) => {
  return text.replace(
    /(```[\s\S]*?```|`[^`\n]+`)|(\[(?:C\d+|\d+|\^\d+|doc[-_][\w-]+|chunk[-_][\w-]+)\])/gi,
    (_match, code, citation) => {
      if (code) return code;
      return `[${citation}](#citation:${encodeURIComponent(citation)})`;
    },
  );
};

console.log("=================================================");
console.log("Starting QA.tsx Citation Resolution Stress Tests");
console.log("=================================================");

let passCount = 0;
let testCount = 0;

function runTest(name, fn) {
  testCount++;
  try {
    fn();
    passCount++;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err.message}`);
    throw err;
  }
}

// ----------------------------------------------------
// Test Group 1: Empty, Null, Undefined Citation Arrays
// ----------------------------------------------------
console.log("\nGroup 1: Empty and Null/Undefined Citation Arrays");

runTest("Empty citation array returns undefined", () => {
  const result = resolveCitation("[C1]", []);
  assert.strictEqual(result, undefined);
});

runTest("Undefined citation array returns undefined", () => {
  const result = resolveCitation("[C1]", undefined);
  assert.strictEqual(result, undefined);
});

runTest("Null citation array returns undefined", () => {
  const result = resolveCitation("[C1]", null);
  assert.strictEqual(result, undefined);
});

runTest("Array of empty objects handles index fallback safely", () => {
  const result = resolveCitation("[C1]", [{}, { citation_id: undefined }]);
  assert.strictEqual(result.citation_id, "[C1]");
  assert.strictEqual(result.chunk_id, "");
  assert.strictEqual(result.evidence_quote, "");
});

// ----------------------------------------------------
// Test Group 2: Out of Bounds Citation IDs ([999], [0], large numbers)
// ----------------------------------------------------
console.log("\nGroup 2: Out-of-bounds Citation IDs");

const sampleCitations = [
  { citation_id: "[C1]", chunk_id: "chunk-1", evidence_quote: "Fact 1" },
  { citation_id: "[C2]", chunk_id: "chunk-2", evidence_quote: "Fact 2" },
];

runTest("Citation ID [999] exceeding list length returns undefined", () => {
  const result = resolveCitation("[999]", sampleCitations);
  assert.strictEqual(result, undefined);
});

runTest("Citation ID [C999] exceeding list length returns undefined", () => {
  const result = resolveCitation("[C999]", sampleCitations);
  assert.strictEqual(result, undefined);
});

runTest("Citation ID [0] returns undefined (1-based index 0 is invalid)", () => {
  const result = resolveCitation("[0]", sampleCitations);
  assert.strictEqual(result, undefined);
});

runTest("Massive number [999999999999999] returns undefined", () => {
  const result = resolveCitation("[999999999999999]", sampleCitations);
  assert.strictEqual(result, undefined);
});

// ----------------------------------------------------
// Test Group 3: Negative numbers, non-numeric strings, weird characters
// ----------------------------------------------------
console.log("\nGroup 3: Negative numbers, non-numeric strings, weird characters");

runTest("Negative numbers [-1], [-99] return undefined", () => {
  assert.strictEqual(resolveCitation("[-1]", sampleCitations), undefined);
  assert.strictEqual(resolveCitation("[-99]", sampleCitations), undefined);
});

runTest("Non-numeric strings return undefined", () => {
  assert.strictEqual(resolveCitation("[invalid]", sampleCitations), undefined);
  assert.strictEqual(resolveCitation("[test]", sampleCitations), undefined);
  assert.strictEqual(resolveCitation("foobar", sampleCitations), undefined);
});

runTest("Hostile/injection characters return undefined safely", () => {
  assert.strictEqual(resolveCitation("[<script>alert(1)</script>]", sampleCitations), undefined);
  assert.strictEqual(resolveCitation("[' OR 1=1 --]", sampleCitations), undefined);
  assert.strictEqual(resolveCitation("[\u0000\uFFFF]", sampleCitations), undefined);
  assert.strictEqual(resolveCitation("[${process.env.SECRET}]", sampleCitations), undefined);
});

// ----------------------------------------------------
// Test Group 4: Non-standard formats [C10], [P-1], [^2], [doc_42]
// ----------------------------------------------------
console.log("\nGroup 4: Non-standard citation formats");

const tenCitations = Array.from({ length: 15 }, (_, i) => ({
  citation_id: `[C${i + 1}]`,
  chunk_id: `chunk-uuid-${i + 1}`,
  document_id: `doc-${i + 1}`,
  evidence_quote: `Quote ${i + 1}`,
}));

runTest("[C10] correctly resolves to 10th item", () => {
  const result = resolveCitation("[C10]", tenCitations);
  assert.ok(result);
  assert.strictEqual(result.chunk_id, "chunk-uuid-10");
});

runTest("[^2] correctly resolves to 2nd item (C2)", () => {
  const result = resolveCitation("[^2]", tenCitations);
  assert.ok(result);
  assert.strictEqual(result.chunk_id, "chunk-uuid-2");
});

runTest("[P-1] resolves if citation_id or chunk_id matches explicitly", () => {
  const customCitations = [
    { citation_id: "[P-1]", chunk_id: "chunk-p1", evidence_quote: "Passage 1 quote" },
  ];
  const result = resolveCitation("[P-1]", customCitations);
  assert.ok(result);
  assert.strictEqual(result.chunk_id, "chunk-p1");
});

runTest("[P-1] returns undefined if no matching citation exists", () => {
  const result = resolveCitation("[P-1]", tenCitations);
  assert.strictEqual(result, undefined);
});

runTest("[doc_42] matches document_id or citation_id", () => {
  const docCitations = [
    { citation_id: "[doc_42]", chunk_id: "chunk-42", document_id: "doc_42", evidence_quote: "Doc 42" },
  ];
  const result = resolveCitation("[doc_42]", docCitations);
  assert.ok(result);
  assert.strictEqual(result.document_id, "doc_42");
});

// ----------------------------------------------------
// Test Group 5: Preprocessing and Regex Pattern Matching
// ----------------------------------------------------
console.log("\nGroup 5: Preprocessing & Markdown Tokenization");

runTest("Multiple citations on the same line are all converted", () => {
  const input = "Here is [C1] and [C2] alongside [^3] and [doc-troubleshooting].";
  const processed = preprocessAnswer(input);
  assert.ok(processed.includes("[[C1]](#citation:%5BC1%5D)"));
  assert.ok(processed.includes("[[C2]](#citation:%5BC2%5D)"));
  assert.ok(processed.includes("[[^3]](#citation:%5B%5E3%5D)"));
  assert.ok(processed.includes("[[doc-troubleshooting]](#citation:%5Bdoc-troubleshooting%5D)"));
});

runTest("Adjacent citations without space are both converted", () => {
  const input = "Dual citations [C1][C2].";
  const processed = preprocessAnswer(input);
  assert.ok(processed.includes("[[C1]](#citation:%5BC1%5D)[[C2]](#citation:%5BC2%5D)"));
});

runTest("Citations inside inline code are NOT converted", () => {
  const input = "Look at `[C1]` in code, but [C2] in prose.";
  const processed = preprocessAnswer(input);
  assert.ok(processed.includes("`[C1]`"), "Inline code must remain untouched");
  assert.ok(!processed.includes("[`[C1]`]"));
  assert.ok(processed.includes("[[C2]](#citation:%5BC2%5D)"));
});

runTest("Citations inside multiline fenced code blocks are NOT converted", () => {
  const input = "```python\n# Reference [C1]\nx = [1]\n```\nOutside [C2].";
  const processed = preprocessAnswer(input);
  assert.ok(processed.includes("```python\n# Reference [C1]\nx = [1]\n```"));
  assert.ok(processed.includes("[[C2]](#citation:%5BC2%5D)"));
});

runTest("Nested brackets like [[C1]] parse inner token cleanly", () => {
  const input = "Nested [[C1]] test";
  const processed = preprocessAnswer(input);
  assert.ok(processed.includes("[[[C1]](#citation:%5BC1%5D)]"));
});

runTest("Empty brackets [] are not converted", () => {
  const input = "Empty [] test";
  const processed = preprocessAnswer(input);
  assert.strictEqual(processed, input);
});

// ----------------------------------------------------
// Test Group 6: Large-Scale and Stress Invariance
// ----------------------------------------------------
console.log("\nGroup 6: Large-Scale Stress Invariance");

runTest("10,000 random queries executed in < 200ms", () => {
  const queries = [
    "[C1]", "[C2]", "[999]", "[-1]", "[^2]", "[doc-1]", "[doc_42]",
    "[invalid]", "[C1000]", "[P-1]", "", "[]", "[NaN]"
  ];
  const t0 = performance.now();
  for (let i = 0; i < 10000; i++) {
    const q = queries[i % queries.length];
    resolveCitation(q, tenCitations);
  }
  const elapsed = performance.now() - t0;
  console.log(`    (10,000 resolves took ${elapsed.toFixed(2)}ms)`);
  assert.ok(elapsed < 200, `Execution too slow: ${elapsed}ms`);
});

console.log("\n=================================================");
console.log(`ALL ${passCount}/${testCount} STRESS TESTS PASSED`);
console.log("=================================================");

