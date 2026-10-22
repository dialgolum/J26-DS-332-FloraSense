import { ShieldPlus, ScanLine } from "lucide-react";
import { useState } from "react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import ImageUpload from "../../components/ui/ImageUpload.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import IntegrationNotice from "../../components/ui/IntegrationNotice.jsx";

export default function DiseasePage() {
  const [file, setFile] = useState(null);
  return <>
    <PageIntro eyebrow="C2 · Plant health" title="Plant disease detection" description="Prepare an image for the disease detection component. Its model, preprocessing, and response contract are not available in the repository yet." action={<StatusBadge tone="pending" dot>Backend integration pending</StatusBadge>} />
    <div className="grid items-start gap-5 xl:grid-cols-[1.1fr_.9fr]"><Card className="p-5 sm:p-6"><ImageUpload label="Plant or leaf image" onFileChange={setFile} /><div className="mt-4 flex justify-end"><Button disabled={!file} title="Connect the C2 endpoint to enable detection"><ScanLine size={16} />Check for disease</Button></div></Card><Card className="p-5 sm:p-6"><div className="mb-5 flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><ShieldPlus size={19} /></span><div><p className="font-display font-extrabold text-forest">Detection result</p><p className="text-xs text-muted">No analysis has been run</p></div></div><IntegrationNotice>{file ? "Image preview is ready. No disease result has been generated." : "Choose an image to preview it here. Disease detection is not connected."}</IntegrationNotice><div className="mt-5 rounded-xl border border-dashed border-line px-4 py-8 text-center"><p className="text-sm font-bold text-muted">Result details will appear here</p><p className="mt-1 text-xs text-muted">Disease classes and confidence fields await the C2 API contract.</p></div></Card></div>
  </>;
}
