import { ScanSearch, Sparkles } from "lucide-react";
import { useState } from "react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import ImageUpload from "../../components/ui/ImageUpload.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import IntegrationNotice from "../../components/ui/IntegrationNotice.jsx";

export default function IdentificationPage() {
  const [file, setFile] = useState(null);
  return <>
    <PageIntro eyebrow="C1 · Plant identification" title="Identify an ornamental plant" description="Upload a clear plant image to prepare an identification request. The classifier API is not connected yet." action={<StatusBadge tone="pending" dot>Backend integration pending</StatusBadge>} />
    <div className="grid items-start gap-5 xl:grid-cols-[1.1fr_.9fr]">
      <Card className="p-5 sm:p-6"><ImageUpload onFileChange={setFile} /><div className="mt-4 flex justify-end"><Button disabled={!file} title="Connect the C1 endpoint to enable identification"><Sparkles size={16} />Identify plant</Button></div></Card>
      <Card className="p-5 sm:p-6"><div className="mb-5 flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><ScanSearch size={19} /></span><div><p className="font-display font-extrabold text-forest">Prediction</p><p className="text-xs text-muted">Model output will appear here</p></div></div><IntegrationNotice>{file ? "Image preview is ready. No identification request has been sent and no prediction has been generated." : "Choose an image to preview it here. The C1 endpoint is not available yet."}</IntegrationNotice><div className="mt-5 grid grid-cols-2 gap-3"><div className="rounded-xl bg-[#f7f9f6] p-4"><p className="text-[11px] font-semibold uppercase tracking-wide text-muted">Variety</p><p className="mt-2 text-sm font-semibold text-muted">Awaiting API</p></div><div className="rounded-xl bg-[#f7f9f6] p-4"><p className="text-[11px] font-semibold uppercase tracking-wide text-muted">Confidence</p><p className="mt-2 text-sm font-semibold text-muted">Awaiting API</p></div></div></Card>
    </div>
  </>;
}
