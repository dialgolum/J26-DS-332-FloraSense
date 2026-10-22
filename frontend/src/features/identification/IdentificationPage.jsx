import { ScanSearch, Sparkles } from "lucide-react";
import { useState } from "react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import ImageUpload from "../../components/ui/ImageUpload.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import IntegrationNotice from "../../components/ui/IntegrationNotice.jsx";
import { identifyPlant } from "../../api/identify.js";
import ChatPanel from "./ChatPanel.jsx";

const nice = (s) => s.replace(" | ", " - ").replace(/_/g, " ");

export default function IdentificationPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  function handleFileChange(f) {
    setFile(f);
    setResult(null);
    setError("");
  }

  async function handleIdentify() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await identifyPlant(file));
    } catch {
      setError("Could not reach the identification service. Check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  const pct = result ? (result.confidence * 100).toFixed(0) : null;
  const notice = error
    ? error
    : loading
    ? "Identifying the plant..."
    : result
    ? result.is_unknown
      ? "I am not sure about this plant. It may be a variety that is not registered yet."
      : `This looks like ${nice(result.label)} (${pct}% confident).`
    : file
    ? "Image preview is ready. Click Identify plant to send it to the classifier."
    : "Choose an image to preview it here.";

  return <>
    <PageIntro
      eyebrow="C1 · Plant identification"
      title="Identify an ornamental plant"
      description="Upload a clear plant image and the C1 classifier will identify the variety."
      action={<StatusBadge tone={result ? "success" : "pending"} dot>{result ? "Prediction ready" : "Waiting for image"}</StatusBadge>}
    />
    <div className="grid items-start gap-5 xl:grid-cols-[1.1fr_.9fr]">
      <Card className="p-5 sm:p-6">
        <ImageUpload onFileChange={handleFileChange} />
        <div className="mt-4 flex justify-end">
          <Button disabled={!file || loading} onClick={handleIdentify}>
            <Sparkles size={16} />{loading ? "Identifying..." : "Identify plant"}
          </Button>
        </div>
      </Card>
      <Card className="p-5 sm:p-6">
        <div className="mb-5 flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><ScanSearch size={19} /></span>
          <div>
            <p className="font-display font-extrabold text-forest">Prediction</p>
            <p className="text-xs text-muted">Model output will appear here</p>
          </div>
        </div>
        <IntegrationNotice>{notice}</IntegrationNotice>
        <div className="mt-5 grid grid-cols-2 gap-3">
          <div className="rounded-xl bg-[#f7f9f6] p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">Variety</p>
            <p className={`mt-2 text-sm font-semibold ${result ? "text-forest" : "text-muted"}`}>
              {result ? (result.is_unknown ? "Unknown variety" : nice(result.label)) : "Awaiting image"}
            </p>
          </div>
          <div className="rounded-xl bg-[#f7f9f6] p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">Confidence</p>
            <p className={`mt-2 text-sm font-semibold ${result ? "text-forest" : "text-muted"}`}>
              {result ? `${pct}%` : "Awaiting image"}
            </p>
          </div>
        </div>
        {result && (
          <div className="mt-3 rounded-xl bg-[#f7f9f6] p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">
              {result.is_unknown ? "Closest matches" : "Other possibilities"}
            </p>
            <ul className="mt-2 space-y-1 text-sm text-muted">
              {(result.is_unknown ? result.top3 : result.top3.slice(1)).map(([name, p]) => (
                <li key={name}>{nice(name)} - {(p * 100).toFixed(0)}%</li>
              ))}
            </ul>
          </div>
        )}
      </Card>
      </div>
    <div className="mt-5">
      <ChatPanel variety={result && !result.is_unknown ? result.label : null} />
    </div>
  </>;
}