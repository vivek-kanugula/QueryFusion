import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Diagnostics from "./views/Diagnostics";
import Documents from "./views/Documents";
import Evaluation from "./views/Evaluation";
import QA from "./views/QA";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Documents />} />
            <Route path="qa" element={<QA />} />
            <Route path="diagnostics" element={<Diagnostics />} />
            <Route path="eval" element={<Evaluation />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
