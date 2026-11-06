"""In-memory trace store with capacity-based eviction."""

from __future__ import annotations

import threading
from collections import OrderedDict

from hybrid_rag.schemas.pipeline import QueryTrace


class TraceStore:
    def __init__(self, max_capacity: int = 500) -> None:
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, QueryTrace] = OrderedDict()
        self._max_capacity = max_capacity

    def save(self, trace: QueryTrace) -> None:
        with self._lock:
            if trace.trace_id in self._traces:
                self._traces.move_to_end(trace.trace_id)
                self._traces[trace.trace_id] = trace
            else:
                self._traces[trace.trace_id] = trace
                while len(self._traces) > self._max_capacity:
                    self._traces.popitem(last=False)

    def get(self, trace_id: str) -> QueryTrace | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                self._traces.move_to_end(trace_id)
            return trace

    def count(self) -> int:
        with self._lock:
            return len(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


trace_store = TraceStore()
