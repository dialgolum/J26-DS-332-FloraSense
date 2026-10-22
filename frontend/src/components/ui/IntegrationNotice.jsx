import { PlugZap } from "lucide-react";

export default function IntegrationNotice({ children = "Backend integration is pending. No prediction has been generated." }) {
  return <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50/80 p-4 text-sm text-amber-950"><PlugZap size={18} className="mt-0.5 shrink-0 text-amber-700" /><div><p className="font-bold">Not connected to a model API</p><p className="mt-1 leading-5 text-amber-900/80">{children}</p></div></div>;
}
