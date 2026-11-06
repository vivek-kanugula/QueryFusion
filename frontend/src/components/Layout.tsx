import {
  BarChart3,
  FileText,
  MessageSquare,
  Search,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

const NAV = [
  { to: "/", icon: FileText, label: "Documents" },
  { to: "/qa", icon: MessageSquare, label: "Q&A" },
  { to: "/diagnostics", icon: Search, label: "Diagnostics" },
  { to: "/eval", icon: BarChart3, label: "Evaluation" },
];

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <nav className="w-56 shrink-0 border-r border-[var(--color-border)] bg-[var(--color-bg-secondary)] flex flex-col">
        <div className="px-4 py-5 border-b border-[var(--color-border)]">
          <h1 className="text-lg font-semibold tracking-tight">Hybrid RAG</h1>
          <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
            Intelligence Platform
          </p>
        </div>
        <div className="flex-1 py-2">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                  isActive
                    ? "bg-[var(--color-primary)]/10 text-[var(--color-primary)] font-medium"
                    : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]"
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
