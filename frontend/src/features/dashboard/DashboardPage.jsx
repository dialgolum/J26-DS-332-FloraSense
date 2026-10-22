import { ArrowDownRight, ArrowUpRight, ArrowRight, Leaf, ScanSearch, ShieldPlus, Activity, MessageCircle } from "lucide-react";
import { Link } from "react-router-dom";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import IntegrationNotice from "../../components/ui/IntegrationNotice.jsx";

const components = [
  { name: "Plant identification", code: "C1 · Vision", text: "Identify ornamental varieties from an image.", to: "/c1/identify", icon: ScanSearch, status: "Frontend ready" },
  { name: "Disease detection", code: "C2 · Plant health", text: "Review disease detection once its model API is available.", to: "/c2/disease", icon: ShieldPlus, status: "Integration pending" },
  { name: "Quality assessment", code: "C3 · Export readiness", text: "Explore buyer-specific sample assessment workflows.", to: "/c3/quality", icon: Leaf, status: "Backend pending" },
  { name: "Supply & demand", code: "C4 · Forecasting", text: "A shared workspace for the forecasting research component.", to: "/c4/forecast", icon: Activity, status: "Integration pending" },
];

function ReadinessRow({ icon: Icon, title, detail, state }) {
  return <div className="flex items-start gap-3 border-b border-line py-4 last:border-0"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-mint text-leaf"><Icon size={17} /></span><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-bold text-ink">{title}</p><StatusBadge tone={state === "ready" ? "ready" : "pending"}>{state === "ready" ? "UI ready" : "API pending"}</StatusBadge></div><p className="mt-1 text-xs leading-5 text-muted">{detail}</p></div></div>;
}

export default function DashboardPage() {
  return <>
    <PageIntro eyebrow="Research workspace" title="Good research starts with a clear view." description="A shared home for FloraSense’s four floriculture research components. Choose a workspace to get started." />
    <div className="mb-7 grid gap-4 sm:grid-cols-3">
      {[{ label: "Research components", value: "04", note: "Across the FloraSense project", icon: Leaf }, { label: "Connected model APIs", value: "00", note: "Backend integrations are pending", icon: Activity }, { label: "Workspace status", value: "Prototype", note: "Frontend foundation", icon: ArrowUpRight }].map(({ label, value, note, icon: Icon }, index) => <Card key={label} className="p-5"><div className="flex items-start justify-between"><div><p className="text-xs font-semibold text-muted">{label}</p><p className={`mt-3 font-display font-extrabold tracking-tight text-forest ${index === 2 ? "text-xl" : "text-3xl"}`}>{value}</p></div><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><Icon size={19} /></span></div><p className="mt-2 text-xs text-muted">{note}</p></Card>)}
    </div>
    <div className="mb-3 flex items-center justify-between"><div><p className="text-xs font-bold uppercase tracking-[.15em] text-leaf">Your workspaces</p><h3 className="mt-1 font-display text-xl font-extrabold text-forest">Research components</h3></div><span className="text-xs font-medium text-muted">Select a component to continue</span></div>
    <div className="grid gap-4 md:grid-cols-2">{components.map(({ name, code, text, to, icon: Icon, status }, index) => <Link key={name} to={to} className="group block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-leaf"><Card className="h-full p-5 transition duration-200 group-hover:-translate-y-0.5 group-hover:border-leaf/30 group-hover:shadow-md sm:p-6"><div className="flex items-start justify-between"><span className={`grid h-12 w-12 place-items-center rounded-2xl ${index === 2 ? "bg-mint text-forest" : "bg-[#f2f6f1] text-leaf"}`}><Icon size={22} /></span><StatusBadge tone={index === 2 ? "pending" : "pending"}>{status}</StatusBadge></div><p className="mt-5 text-[10px] font-extrabold uppercase tracking-[.16em] text-leaf">{code}</p><h4 className="mt-1 font-display text-lg font-extrabold text-forest">{name}</h4><p className="mt-2 max-w-md text-sm leading-6 text-muted">{text}</p><div className="mt-5 flex items-center gap-2 text-xs font-bold text-forest">Open workspace <ArrowRight size={15} className="transition group-hover:translate-x-1" /></div></Card></Link>)}</div>
    <div className="mt-7 grid gap-4 lg:grid-cols-[1.35fr_.85fr]">
      <Card className="p-5 sm:p-6"><div className="mb-1 flex items-center justify-between"><div><p className="text-xs font-bold uppercase tracking-[.15em] text-leaf">Build status</p><h3 className="mt-1 font-display text-lg font-extrabold text-forest">A shared foundation, ready to grow</h3></div><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><ArrowDownRight size={19} /></span></div><ReadinessRow icon={ScanSearch} title="Shared frontend shell" detail="Responsive navigation, routing, reusable UI, and feature-based folders." state="ready" /><ReadinessRow icon={MessageCircle} title="Research APIs" detail="Each feature has a versioned API seam; no requests are made yet." state="pending" /></Card>
      <div className="space-y-4"><IntegrationNotice>Pages for C1, C2, and C4 are deliberately clear of simulated AI outputs. Connect a real service before displaying results.</IntegrationNotice><Card className="p-5"><p className="text-xs font-bold uppercase tracking-[.15em] text-leaf">API convention</p><p className="mt-2 text-sm font-bold text-forest">Versioned service contracts</p><code className="mt-3 block rounded-xl bg-[#f4f7f3] px-3 py-2.5 text-xs text-forest">/api/v1/&lt;component&gt;/...</code><p className="mt-2 text-xs leading-5 text-muted">Base URL is configured with <code>VITE_API_BASE_URL</code>.</p></Card></div>
    </div>
  </>;
}
