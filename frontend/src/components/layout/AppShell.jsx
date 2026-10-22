import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import apiClient from "../../api/apiClient.js";
import {
  Activity, BarChart3, Leaf, Menu, MessageCircle, ScanSearch,
  ShieldPlus, Sprout, X,
} from "lucide-react";

const groups = [
  { label: "Workspace", links: [{ to: "/", title: "Overview", icon: BarChart3, end: true }] },
  { label: "Research components", links: [
    { to: "/c1/identify", title: "Plant identification", icon: ScanSearch, tag: "C1" },
    { to: "/c1/chat", title: "Plant assistant", icon: MessageCircle, tag: "C1" },
    { to: "/c2/disease", title: "Disease detection", icon: ShieldPlus, tag: "C2" },
    { to: "/c3/quality", title: "Quality assessment", icon: Leaf, tag: "C3" },
    { to: "/c4/forecast", title: "Supply & demand", icon: Activity, tag: "C4" },
  ] },
];

function Sidebar({ onNavigate, operational }) {
  return (
    <div className="flex h-full flex-col px-4 py-5">
      <NavLink to="/" onClick={onNavigate} className="mb-9 flex items-center gap-3 px-2">
        <span className="grid h-10 w-10 place-items-center rounded-2xl bg-forest text-white shadow-sm"><Sprout size={22} /></span>
        <span><span className="block font-display text-lg font-extrabold tracking-tight text-forest">FloraSense</span><span className="block text-[11px] font-semibold uppercase tracking-[.17em] text-muted">{operational ? "Plant quality control" : "Research workspace"}</span></span>
      </NavLink>
      <nav className="space-y-7">
        {groups.map((group) => <div key={group.label}>
          <p className="mb-2 px-3 text-[10px] font-bold uppercase tracking-[.16em] text-muted">{operational && group.label === "Research components" ? "Services" : group.label}</p>
          <div className="space-y-1">{group.links.map(({ to, title, icon: Icon, tag, end }) => <NavLink
            key={to} to={to} end={end} onClick={onNavigate}
            className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}
          ><Icon size={18} strokeWidth={1.8} /><span className="flex-1">{title}</span>{tag && <span className="nav-tag">{tag}</span>}</NavLink>)}</div>
        </div>)}
      </nav>
      <div className="mt-auto rounded-2xl border border-line bg-white/70 p-4">
        <div className="mb-2 flex items-center gap-2 text-xs font-bold text-forest"><span className="h-2 w-2 rounded-full bg-amber-400" /> {operational ? "Plant inspections" : "Prototype mode"}</div>
        <p className="text-xs leading-5 text-muted">{operational ? "Select a plant and buyer requirement to begin an inspection." : "Select a service to view its available tools."}</p>
      </div>
    </div>
  );
}

const pageNames = {
  "/": "Overview", "/c1/identify": "Plant identification", "/c1/chat": "Plant assistant",
  "/c2/disease": "Disease detection", "/c3/quality": "Quality assessment", "/c4/forecast": "Supply & demand",
};

export default function AppShell() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const operational = location.pathname === "/c3/quality";
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    if (!operational) return undefined;
    let active = true;
    const controller = new AbortController();
    setApiStatus("checking");
    const relevant = (config) => config?.url?.startsWith("/api/v1/c3");
    const interceptor = apiClient.interceptors.response.use((response) => {
      if (active && relevant(response.config)) setApiStatus("connected");
      return response;
    }, (error) => {
      if (active && relevant(error.config) && error.code !== "ERR_CANCELED") {
        setApiStatus(error.response && error.response.status < 500 ? "connected" : "unavailable");
      }
      return Promise.reject(error);
    });
    const checkConnection = () => apiClient.get("/health", { signal: controller.signal }).then((response) => {
      if (active) setApiStatus(response.data?.data?.status === "ok" ? "connected" : "unavailable");
    }).catch(() => { if (active) setApiStatus("unavailable"); });
    checkConnection();
    const timer = setInterval(checkConnection, 30_000);
    return () => { active = false; controller.abort(); clearInterval(timer); apiClient.interceptors.response.eject(interceptor); };
  }, [operational]);
  return <div className="min-h-screen bg-canvas text-ink">
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-[258px] border-r border-line bg-[#f9fbf8] lg:block"><Sidebar operational={operational} /></aside>
    {menuOpen && <div className="fixed inset-0 z-40 lg:hidden">
      <button aria-label="Close navigation" onClick={() => setMenuOpen(false)} className="absolute inset-0 bg-forest/30" />
      <aside className="relative h-full w-[280px] bg-[#f9fbf8] shadow-2xl"><div className="absolute right-3 top-4"><button className="icon-button" onClick={() => setMenuOpen(false)} aria-label="Close menu"><X size={19} /></button></div><Sidebar operational={operational} onNavigate={() => setMenuOpen(false)} /></aside>
    </div>}
    <div className="lg:pl-[258px]">
      <header className="sticky top-0 z-20 border-b border-line/80 bg-canvas/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[72px] max-w-[1500px] items-center justify-between px-4 sm:px-7 xl:px-10">
          <div className="flex items-center gap-3"><button className="icon-button lg:hidden" onClick={() => setMenuOpen(true)} aria-label="Open menu"><Menu size={20} /></button><div><p className="text-[11px] font-semibold uppercase tracking-[.13em] text-muted">Floriculture intelligence</p><h1 className="font-display text-base font-bold text-forest">{pageNames[location.pathname] || "Workspace"}</h1></div></div>
          <div className="flex items-center gap-3"><span className="hidden items-center gap-2 rounded-full border border-line bg-white px-3 py-2 text-xs font-semibold text-muted sm:flex"><span className={`h-2 w-2 rounded-full ${operational && apiStatus === "connected" ? "bg-emerald-500" : "bg-amber-400"}`} />{operational ? { checking: "Checking connection", connected: "API connected", unavailable: "API unavailable" }[apiStatus] : "API status not checked"}</span><div className="grid h-9 w-9 place-items-center rounded-full bg-mint text-sm font-bold text-forest">FS</div></div>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-4 py-7 sm:px-7 sm:py-9 xl:px-10"><Outlet /></main>
    </div>
  </div>;
}
