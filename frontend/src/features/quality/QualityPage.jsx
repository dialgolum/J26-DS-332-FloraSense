import { ClipboardCheck, Leaf } from "lucide-react";
import { useState } from "react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import Field, { Select } from "../../components/ui/Field.jsx";
import ImageUpload from "../../components/ui/ImageUpload.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import IntegrationNotice from "../../components/ui/IntegrationNotice.jsx";

// Snapshot values from the current C3 workbook for form layout only. Replace
// these options with the C3 reference-data API before using this operationally.
const sampleIds = Array.from({ length: 19 }, (_, index) => `S${String(index + 1).padStart(3, "0")}`);

export default function QualityPage() {
  const [plant, setPlant] = useState("P01");
  const [requirement, setRequirement] = useState("R001");
  const [sample, setSample] = useState("S001");
  const [file, setFile] = useState(null);
  return <>
    <PageIntro eyebrow="C3 · Export readiness" title="Buyer-specific quality assessment" description="Select a plant, buyer requirement, and sample to prepare an assessment. Results will be available when the C3 API is connected." action={<StatusBadge tone="pending" dot>Backend integration pending</StatusBadge>} />
    <div className="grid items-start gap-5 xl:grid-cols-[.85fr_1.15fr]">
      <Card className="p-5 sm:p-6"><div className="mb-5 flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-forest"><ClipboardCheck size={19} /></span><div><p className="font-display font-extrabold text-forest">Assessment setup</p><p className="text-xs text-muted">Reference options shown for layout only</p></div></div><div className="space-y-4"><Field label="Plant"><Select value={plant} onChange={(event) => setPlant(event.target.value)}><option value="P01">P01 · Aglaonema silver queen</option></Select></Field><Field label="Buyer requirement"><Select value={requirement} onChange={(event) => setRequirement(event.target.value)}><option value="R001">R001 · B001</option><option value="R002">R002 · B002</option></Select></Field><Field label="Sample"><Select value={sample} onChange={(event) => setSample(event.target.value)}>{sampleIds.map((id) => <option key={id}>{id}</option>)}</Select></Field></div><div className="mt-5"><IntegrationNotice>Selectors use a local workbook snapshot and have not queried C3. No assessment or AI result is available from this page yet.</IntegrationNotice></div><Button className="mt-5 w-full" disabled title="Connect the C3 assessment endpoint to enable assessment"><ClipboardCheck size={16} />Run assessment</Button></Card>
      <div className="space-y-5"><Card className="p-5 sm:p-6"><div className="mb-4 flex items-center gap-2"><Leaf size={17} className="text-leaf" /><h3 className="font-display font-extrabold text-forest">Sample image</h3></div><ImageUpload label="Assessment image preview" onFileChange={setFile} /></Card><Card className="p-5 sm:p-6"><div className="mb-4 flex items-center justify-between"><div><h3 className="font-display font-extrabold text-forest">Assessment results</h3><p className="mt-1 text-xs text-muted">{sample} · {requirement}</p></div><StatusBadge tone="neutral">Awaiting API</StatusBadge></div><div className="rounded-xl border border-dashed border-line px-4 py-7 text-center"><p className="text-sm font-bold text-muted">No assessment has been run</p><p className="mt-1 text-xs text-muted">{file ? "The selected image is available for preview." : "Select an image and connect the C3 endpoint to see checks and reasons."}</p></div></Card></div>
    </div>
  </>;
}
