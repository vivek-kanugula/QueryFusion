import {
  BarChart3,
  FileText,
  MessageSquare,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

const NAV = [
  { to: "/", icon: FileText, label: "Documents" },
  { to: "/qa", icon: MessageSquare, label: "Q&A" },
  { to: "/diagnostics", icon: Search, label: "Diagnostics" },
  { to: "/eval", icon: BarChart3, label: "Evaluation" },
];

export default function Layout() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("theme");
      if (saved === "light" || saved === "dark") return saved;
      return window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    return "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <div className="flex h-screen overflow-hidden">
      <nav className="w-[248px] shrink-0 border-r border-[var(--border)] bg-[var(--surface-deep)] flex flex-col">
        <div className="px-5 h-14 flex flex-col justify-center border-b border-[var(--border)]">
          <h1 className="text-[15px] font-medium tracking-tight text-[var(--text-primary)]">
            Hybrid RAG
          </h1>
          <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
            Retrieval benchmarking
          </p>
        </div>

        <div className="flex-1 py-3 px-3">
          <div className="flex flex-col gap-0.5">
            {NAV.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 px-3 h-9 text-[13px] rounded-lg transition-colors ${
                    isActive
                      ? "bg-[var(--surface-raised)] text-[var(--text-primary)] font-medium"
                      : "text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)]"
                  }`
                }
              >
                <Icon size={16} strokeWidth={1.5} />
                {label}
              </NavLink>
            ))}
          </div>
        </div>

        <div className="px-3 pb-3">
          <button
            type="button"
            onClick={() => setTheme((p) => (p === "dark" ? "light" : "dark"))}
            className="w-full flex items-center gap-2.5 px-3 h-9 text-[13px] rounded-lg text-[var(--text-muted)] hover:bg-[var(--surface-raised)] hover:text-[var(--text-secondary)] transition-colors cursor-pointer"
          >
            {theme === "dark" ? (
              <Sun size={16} strokeWidth={1.5} />
            ) : (
              <Moon size={16} strokeWidth={1.5} />
            )}
            <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
          </button>
        </div>
      </nav>

      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
