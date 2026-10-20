import { LoaderCircle } from "lucide-react";

export default function LoadingIndicator({ label = "Loading", compact = false }) {
  return <div className={`flex items-center justify-center gap-2 text-sm font-medium text-muted ${compact ? "py-3" : "py-10"}`} role="status"><LoaderCircle className="animate-spin text-leaf" size={18} />{label}</div>;
}
