"""RSS Memory Spike: measures resident memory with both models loaded via fastembed.

Run: python scripts/rss_spike.py

Validates memory feasibility. Records baseline, post-embedder,
and post-reranker RSS overhead.
"""
import os
import sys
import time


def get_rss_mb() -> float:
    """Get current process RSS in MB. Cross-platform."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):  # noqa: N801
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(pmc)
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024 * 1024)
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main():
    print("=" * 60)
    print("RSS Memory Spike — Container Memory Budget Feasibility Check")
    print("=" * 60)

    baseline = get_rss_mb()
    print(f"\n[1/4] Baseline RSS: {baseline:.1f} MB")

    print("\n[2/4] Loading embedding model (BAAI/bge-small-en-v1.5)...")
    t0 = time.time()
    from fastembed import TextEmbedding
    embedder = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
    )
    list(embedder.embed(["warmup query"]))
    t1 = time.time()
    post_embed = get_rss_mb()
    print(f"   Embedder loaded in {t1 - t0:.1f}s")
    print(f"   RSS after embedder: {post_embed:.1f} MB (+{post_embed - baseline:.1f} MB)")

    print("\n[3/4] Loading reranker model (Xenova/ms-marco-MiniLM-L-6-v2)...")
    t2 = time.time()
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    reranker = TextCrossEncoder(
        model_name="Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
    )
    list(reranker.rerank("test query", ["test passage"], top_k=1))
    t3 = time.time()
    post_rerank = get_rss_mb()
    print(f"   Reranker loaded in {t3 - t2:.1f}s")
    print(f"   RSS after reranker: {post_rerank:.1f} MB (+{post_rerank - post_embed:.1f} MB)")

    total_model_overhead = post_rerank - baseline
    boot_time = t3 - t0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Baseline RSS:        {baseline:.1f} MB")
    print(f"  + Embedder:          {post_embed:.1f} MB  (+{post_embed - baseline:.1f} MB)")
    print(f"  + Reranker:          {post_rerank:.1f} MB  (+{post_rerank - post_embed:.1f} MB)")
    print(f"  Total model overhead: {total_model_overhead:.1f} MB")
    print(f"  Model load time:     {boot_time:.1f}s")
    print()

    container_limit = 512
    headroom = container_limit - post_rerank
    if headroom > 50:
        verdict = f"GO — {headroom:.0f} MB headroom on {container_limit} MB container"
    elif headroom > 0:
        verdict = f"CAUTION — only {headroom:.0f} MB headroom, may OOM under load"
    else:
        verdict = f"NO-GO — exceeds {container_limit} MB by {-headroom:.0f} MB, need larger memory tier or ONNX optimization"

    print(f"  Memory verdict: {verdict}")
    print("=" * 60)

    return post_rerank < container_limit


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
