const styles = {
  ready: "bg-emerald-50 text-emerald-700 ring-emerald-600/10",
  pending: "bg-amber-50 text-amber-800 ring-amber-600/10",
  neutral: "bg-slate-100 text-slate-600 ring-slate-500/10",
  pass: "bg-emerald-50 text-emerald-700 ring-emerald-600/10",
  fail: "bg-rose-50 text-rose-700 ring-rose-600/10",
  review: "bg-amber-50 text-amber-800 ring-amber-600/10",
};

export default function StatusBadge({ children, tone = "pending", dot = false }) {
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-bold ring-1 ring-inset ${styles[tone] || styles.neutral}`}>{dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}{children}</span>;
}
