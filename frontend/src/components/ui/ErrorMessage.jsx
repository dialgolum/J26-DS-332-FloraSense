import { CircleAlert } from "lucide-react";

export default function ErrorMessage({ title = "Something went wrong", children }) {
  return <div role="alert" className="flex gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"><CircleAlert size={18} className="mt-0.5 shrink-0" /><div><p className="font-bold">{title}</p><p className="mt-1 leading-5 text-rose-700">{children}</p></div></div>;
}
