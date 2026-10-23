import { useEffect, useRef, useState } from "react";
import Card from "../../components/ui/Card.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import ErrorMessage from "../../components/ui/ErrorMessage.jsx";
import { getApiErrorMessage, inspectFrontImage, inspectUploadedPlant } from "./api.js";
import { validateFrontImage, validateInspectionImages } from "./uploadValidation.js";

export default function AutomaticLeafUpload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [verified, setVerified] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const request = useRef(null);

  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => request.current?.abort(), []);

  function invalidate() {
    request.current?.abort(); request.current = null;
    setBusy(false); setResult(null); setError("");
  }

  async function inspect() {
    const validationError = validateFrontImage(file);
    if (validationError) { setError(validationError); return; }
    const controller = new AbortController();
    request.current = controller;
    setBusy(true); setResult(null); setError("");
    try {
      const response = await inspectFrontImage(file, verified, controller.signal);
      if (request.current === controller) setResult(response);
    } catch (requestError) {
      if (request.current === controller && !controller.signal.aborted) {
        setError(getApiErrorMessage(requestError, "Image inspection failed."));
      }
    } finally {
      if (request.current === controller) { setBusy(false); request.current = null; }
    }
  }

  return <Card className="mb-5 p-5 sm:p-6">
    <h2 className="font-display text-lg font-extrabold text-forest">New P01 image — automatic leaf prototype</h2>
    <p className="mt-2 text-sm text-muted">Upload a front view containing the whole plant and graph paper. This experimental count is separate from historical workbook annotations and saved height sessions. Uploads are processed in memory and are not saved.</p>
    <div className="mt-4 grid gap-5 lg:grid-cols-2">
      <div className="space-y-4">
        <label className="block text-sm font-bold">Front image (JPEG/PNG, up to 25 MiB)
          <input type="file" accept="image/jpeg,image/png" className="mt-2 block w-full text-sm" onChange={(event) => {
            invalidate();
            const next = event.target.files?.[0] || null;
            const message = next ? validateFrontImage(next) : "";
            setFile(message ? null : next); setVerified(false); setError(message);
          }} />
        </label>
        <p className="text-sm">P01 graph paper: major squares 1 cm ? 1 cm, divided into 5 ? 5 minor squares of 0.2 cm. Calibration requires distinguishing major lines from minor lines.</p>
        <label className="flex items-start gap-2 text-sm"><input type="checkbox" checked={verified} onChange={(event) => { invalidate(); setVerified(event.target.checked); }} />This image uses the confirmed P01 graph-paper spacing above.</label>
        <p className="text-xs text-muted">Calibration requires regular lines in both axes and multiple background areas. No individual leaf heights are produced without reliable leaf endpoints. Leaf counting remains subject to manual review.</p>
        <Button onClick={inspect} disabled={busy || !file}>{busy ? "Inspecting image…" : "Detect leaf regions"}</Button>
        {error && <ErrorMessage title="Upload inspection failed">{error}</ErrorMessage>}
      </div>
      {preview && <figure><img src={preview} alt="New P01 front image selected for automatic inspection" className="max-h-[400px] w-full rounded-xl border border-line object-contain" /><figcaption className="mt-1 text-xs text-muted">Uploaded image: {file?.name}</figcaption></figure>}
    </div>
    {result && <div className="mt-5 border-t border-line pt-5">
      <div className="flex flex-wrap items-center gap-3"><h3 className="font-bold text-forest">Uploaded-image OpenCV result</h3><StatusBadge tone="review">{result.status}</StatusBadge></div>
      <div className="mt-3 grid gap-3 text-sm sm:grid-cols-3">
        <p>Estimated leaf count (unverified): <strong>{result.estimated_count ?? "Unavailable"}</strong></p>
        <p>Detected candidate regions: <strong>{result.detected_region_count}</strong></p>
        <p>Count reliable: <strong>{result.count_reliable ? "Yes" : "No"}</strong></p>
        <p>Confidence: <strong>{result.confidence == null ? "Not statistically calibrated" : result.confidence}</strong></p>
        <p>Calibration: <strong>{result.calibration.status}</strong></p>
        <p>Background scale: <strong>{result.calibration.scale_px_per_cm == null ? "Unavailable" : `${result.calibration.scale_px_per_cm.toFixed(2)} px/cm (processing image)`}</strong></p>
      </div>
      <p className="mt-3 text-sm">Individual leaf heights: unavailable. Buyer quality decision: not generated.</p>
      <ul className="mt-3 list-inside list-disc space-y-1 text-sm text-muted">{result.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
      <p className="mt-3 text-xs text-muted">{result.calibration.limitation}</p>
      <figure className="mt-4"><img src={result.annotated_preview} alt="OpenCV candidate leaf regions numbered for manual review" className="max-h-[650px] w-full rounded-xl border border-line object-contain" /><figcaption className="mt-2 text-xs text-muted">Orange outlines and numbers identify candidate regions, including possible fragments and merged leaves. Coordinates refer to the {result.processing_dimensions.join(" × ")} processing image; original oriented dimensions: {result.native_dimensions.join(" × ")}.</figcaption></figure>
      {result.leaf_regions.length > 0 && <details className="mt-3 text-sm"><summary className="cursor-pointer font-bold">Region details and ambiguity flags</summary><ul className="mt-2 space-y-1">{result.leaf_regions.map((region) => <li key={region.region_id}>Region {region.region_id}: box [{region.bbox_px.join(", ")}]; {region.ambiguity_flags.join(", ") || "No local flag; leaf identity remains unverified"}</li>)}</ul></details>}
    </div>}
  </Card>;
}


function UploadPhoto({ file, view }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!file) { setUrl(""); return undefined; }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url ? <img src={url} alt={`Uploaded ${view.toLowerCase()} view`} className="mt-3 h-52 w-full rounded-xl border border-line object-contain" /> : null;
}

export function UploadPlantInspection({ plantId, requirementId, ready }) {
  const [files, setFiles] = useState({ Front: null, Root: null, Stem: null });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const request = useRef(null);
  useEffect(() => () => request.current?.abort(), []);

  function invalidate() {
    request.current?.abort(); request.current = null;
    setResult(null); setBusy(false); setError("");
  }
  async function inspect() {
    const validation = validateInspectionImages(files);
    if (validation) { setError(validation); return; }
    if (!ready) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true); setResult(null); setError("");
    try {
      const response = await inspectUploadedPlant(plantId, requirementId, files, controller.signal);
      if (request.current === controller) setResult(response);
    } catch (failure) {
      if (request.current === controller && !controller.signal.aborted) setError(getApiErrorMessage(failure, "Inspection failed. Please retry."));
    } finally {
      if (request.current === controller) { setBusy(false); request.current = null; }
    }
  }
  return <>
    <Card className="p-5 sm:p-6">
      <h2 className="font-display text-lg font-extrabold text-forest">Upload plant photographs</h2>
      <p className="mt-2 text-sm text-muted">Front and Root images are required. Stem is optional and reserved for later checks. JPEG/PNG, up to 25 MiB per image. Uploads are temporary.</p>
      <div className="mt-4 grid gap-4 sm:grid-cols-3">{["Front", "Root", "Stem"].map((view) => <div key={view}>
        <label className="block text-sm font-bold">{view} image{view === "Stem" ? " (optional)" : " (required)"}<input type="file" accept="image/jpeg,image/png" className="mt-2 block w-full text-sm" onChange={(event) => {
          invalidate();
          const file = event.target.files?.[0] || null;
          const message = file ? validateFrontImage(file) : "";
          setFiles((current) => ({ ...current, [view]: message ? null : file })); setError(message);
        }} /></label>
        <UploadPhoto file={files[view]} view={view} />
      </div>)}</div>
      {error && <div className="mt-4"><ErrorMessage title="Inspection unavailable">{error}</ErrorMessage></div>}
      <div className="mt-5"><Button disabled={!ready || busy || Boolean(validateInspectionImages(files))} onClick={inspect}>{busy ? "Inspecting plant..." : "Run Inspection"}</Button></div>
      {busy && <p role="status" className="mt-3 text-sm text-muted">Running experimental classifiers...</p>}
    </Card>
    {result && <Card className="p-5 sm:p-6" aria-live="polite">
      <div className="flex items-center justify-between gap-3"><h2 className="font-bold text-forest">Overall buyer assessment result</h2><StatusBadge tone="review">REVIEW</StatusBadge></div>
      <p className="mt-3 text-sm text-muted">Experimental AI predictions require human review. Buyer approval remains pending until required measurements and checks are completed.</p>
      <ul className="mt-4 divide-y divide-line">{Object.entries(result.AI_Observations).map(([key, observation]) => <li key={key} className="py-4">
        <div className="flex items-center justify-between gap-3"><h3 className="text-sm font-bold">{key === "root_B" ? "Root distribution" : "Branch distribution"}</h3><StatusBadge tone="review">{observation.status === "success" ? "AI prediction" : "REVIEW"}</StatusBadge></div>
        <p className="mt-2 text-sm">{observation.status === "success" ? observation.prediction.predicted_label : observation.reason}</p>
      </li>)}{result.Checks.map((check) => <li key={check.Check} className="flex items-center justify-between gap-3 py-4"><h3 className="text-sm font-bold">{check.Check}</h3><StatusBadge tone="pending">PENDING</StatusBadge></li>)}</ul>
    </Card>}
  </>;
}
