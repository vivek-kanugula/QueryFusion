import { useCallback, useEffect, useRef, useState } from "react";
import { type SSEEvent, type StreamQueryOptions, streamQuery } from "../lib/api";

export interface QueryState {
  isStreaming: boolean;
  stages: string[];
  answer: Record<string, unknown> | null;
  verification: Record<string, unknown> | null;
  confidence: Record<string, unknown> | null;
  completed: Record<string, unknown> | null;
  error: string | null;
  traceId: string | null;
}

const INITIAL: QueryState = {
  isStreaming: false,
  stages: [],
  answer: null,
  verification: null,
  confidence: null,
  completed: null,
  error: null,
  traceId: null,
};

export function useSSEQuery(initialState?: Partial<QueryState>) {
  const [state, setState] = useState<QueryState>(() => ({
    ...INITIAL,
    ...initialState,
    isStreaming: false,
  }));
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const execute = useCallback(
    (
      question: string,
      options: StreamQueryOptions,
    ) => {
      controllerRef.current?.abort();
      setState({ ...INITIAL, isStreaming: true });

      const ctrl = streamQuery(
        question,
        options,
        (evt: SSEEvent) => {
          setState((prev) => {
            const d = evt.data as Record<string, unknown>;
            switch (evt.event) {
              case "stage":
                return {
                  ...prev,
                  stages: [...prev.stages, d.name as string],
                  traceId: (d.trace_id as string) ?? prev.traceId,
                };
              case "answer":
                return { ...prev, answer: d };
              case "verification":
                return { ...prev, verification: d };
              case "confidence":
                return { ...prev, confidence: d };
              case "completed":
                return {
                  ...prev,
                  completed: d,
                  traceId: (d.trace_id as string) ?? prev.traceId,
                  isStreaming: false,
                };
              case "error":
                return {
                  ...prev,
                  error: (d.message as string) ?? "Unknown error",
                  isStreaming: false,
                };
              default:
                return prev;
            }
          });
        },
        () => setState((p) => ({ ...p, isStreaming: false })),
        (err) =>
          setState((p) => ({
            ...p,
            error: err.message,
            isStreaming: false,
          })),
      );
      controllerRef.current = ctrl;
    },
    [],
  );

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    setState((p) => ({ ...p, isStreaming: false }));
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    setState(INITIAL);
  }, []);

  const restore = useCallback((saved: Partial<QueryState>) => {
    controllerRef.current?.abort();
    setState({
      ...INITIAL,
      ...saved,
      isStreaming: false,
    });
  }, []);

  return { ...state, execute, cancel, reset, restore };
}
