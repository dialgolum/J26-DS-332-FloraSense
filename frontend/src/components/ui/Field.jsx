export default function Field({ label, hint, children }) {
  return <div><label className="mb-2 block text-sm font-bold text-ink">{label}</label>{children}{hint && <p className="mt-1.5 text-xs leading-5 text-muted">{hint}</p>}</div>;
}

export function Select({ value, onChange, children, disabled = false, ...props }) {
  return <select value={value} onChange={onChange} disabled={disabled} className="field-control disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-muted" {...props}>{children}</select>;
}
