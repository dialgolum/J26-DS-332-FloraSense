import { useEffect, useMemo, useState } from "react";
import { ClipboardCheck, Image as ImageIcon, Leaf, RefreshCw } from "lucide-react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import Field, { Select } from "../../components/ui/Field.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import LoadingIndicator from "../../components/ui/LoadingIndicator.jsx";
import ErrorMessage from "../../components/ui/ErrorMessage.jsx";
import {
  getApiErrorMessage, getHeightSessions, getImagePreviewUrl, getPlantTypes,
  getRequirements, getSampleImages, getSamples, reevaluateHeightSession,
} from "./api.js";

function statusTone(status) {
  if (status === "PASS") return "pass";
  if (status === "FAIL") return "fail";
  if (status === "MANUAL REVIEW") return "review";
  return "neutral";
}

function formatDate(value) {
  if (!value) return "Date unavailable";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

export default function QualityPage() {
  const [plants, setPlants] = useState([]);
  const [plantId, setPlantId] = useState("");
  const [requirements, setRequirements] = useState([]);
  const [requirementId, setRequirementId] = useState("");
  const [samples, setSamples] = useState([]);
  const [sampleId, setSampleId] = useState("");
  const [images, setImages] = useState([]);
  const [imageId, setImageId] = useState("");
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const [result, setResult] = useState(null);
  const [loadingPlants, setLoadingPlants] = useState(true);
  const [loadingReferences, setLoadingReferences] = useState(false);
  const [loadingSampleData, setLoadingSampleData] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState("");
  const [sampleDataError, setSampleDataError] = useState("");

  useEffect(() => {
    let active = true;
    getPlantTypes().then((items) => {
      if (!active) return;
      setPlants(items);
      setPlantId(items[0]?.Plant_ID || "");
    }).catch((requestError) => {
      if (active) setError(getApiErrorMessage(requestError, "Could not load plant types."));
    }).finally(() => { if (active) setLoadingPlants(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!plantId) {
      setRequirements([]); setRequirementId(""); setSamples([]); setSampleId("");
      return undefined;
    }
    let active = true;
    setLoadingReferences(true);
    setError("");
    setResult(null);
    Promise.all([getRequirements(plantId), getSamples(plantId)]).then(([profiles, sampleRows]) => {
      if (!active) return;
      setRequirements(profiles);
      setSamples(sampleRows);
      setRequirementId((current) => profiles.some((item) => item.Requirement_ID === current)
        ? current : profiles[0]?.Requirement_ID || "");
      setSampleId((current) => sampleRows.some((item) => item.Sample_ID === current)
        ? current : sampleRows[0]?.Sample_ID || "");
    }).catch((requestError) => {
      if (active) setError(getApiErrorMessage(requestError, "Could not load requirements and samples."));
    }).finally(() => { if (active) setLoadingReferences(false); });
    return () => { active = false; };
  }, [plantId]);

  useEffect(() => {
    if (!sampleId) {
      setImages([]); setImageId(""); setSessions([]); setSessionId("");
      return undefined;
    }
    let active = true;
    setLoadingSampleData(true);
    setSampleDataError("");
    setResult(null);
    Promise.all([getSampleImages(sampleId), getHeightSessions(sampleId)]).then(([imageRows, sessionRows]) => {
      if (!active) return;
      setImages(imageRows);
      setSessions(sessionRows);
      setImageId((current) => imageRows.some((item) => item.Image_ID === current)
        ? current : imageRows[0]?.Image_ID || "");
      setSessionId((current) => sessionRows.some((item) => item.session_id === current)
        ? current : sessionRows[0]?.session_id || "");
    }).catch((requestError) => {
      if (active) setSampleDataError(getApiErrorMessage(requestError, "Could not load sample images and height sessions."));
    }).finally(() => { if (active) setLoadingSampleData(false); });
    return () => { active = false; };
  }, [sampleId]);

  const selectedPlant = plants.find((item) => item.Plant_ID === plantId);
  const selectedImage = images.find((item) => item.Image_ID === imageId);
  const selectedRequirement = requirements.find((item) => item.Requirement_ID === requirementId);
  const selectedSession = sessions.find((item) => item.session_id === sessionId);
  const previewUrl = useMemo(() => selectedImage ? getImagePreviewUrl(selectedImage.Image_ID) : "", [selectedImage]);

  async function runReevaluation() {
    if (!sessionId || !requirementId) return;
    setEvaluating(true);
    setError("");
    setResult(null);
    try {
      setResult(await reevaluateHeightSession(sessionId, requirementId));
    } catch (requestError) {
      setError(getApiErrorMessage(requestError, "Could not reevaluate this height session."));
    } finally {
      setEvaluating(false);
    }
  }

  return <>
    <PageIntro eyebrow="C3 · Export readiness" title="Buyer-specific quality assessment" description="Browse workbook-backed plant records and reevaluate a saved P01 height-measurement session against a buyer requirement." action={<StatusBadge tone="ready" dot>C3 API connected</StatusBadge>} />
    {error && <div className="mb-5"><ErrorMessage title="C3 request failed">{error}</ErrorMessage></div>}
    {loadingPlants ? <Card className="p-5"><LoadingIndicator label="Loading plant types…" /></Card> : plants.length === 0 ? <Card className="p-5"><p className="text-sm text-muted">No plant types are available.</p></Card> : <>
      <div className="grid items-start gap-5 xl:grid-cols-[.85fr_1.15fr]">
        <Card className="p-5 sm:p-6">
          <div className="mb-5 flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-forest"><ClipboardCheck size={19} /></span><div><p className="font-display font-extrabold text-forest">Assessment setup</p><p className="text-xs text-muted">Options loaded from the C3 workbook</p></div></div>
          <div className="space-y-4">
            <Field label="Plant type"><Select value={plantId} onChange={(event) => setPlantId(event.target.value)}>{plants.map((plant) => <option key={plant.Plant_ID} value={plant.Plant_ID}>{plant.Plant_ID} · {plant.Plant_Name}</option>)}</Select></Field>
            <Field label="Buyer requirement"><Select value={requirementId} onChange={(event) => { setRequirementId(event.target.value); setResult(null); }} disabled={loadingReferences || requirements.length === 0}>{requirements.map((profile) => <option key={profile.Requirement_ID} value={profile.Requirement_ID}>{profile.Requirement_ID} · {profile.Buyer_ID} · {profile.Min_Height_cm}–{profile.Max_Height_cm} cm</option>)}</Select></Field>
            <Field label="Sample"><Select value={sampleId} onChange={(event) => setSampleId(event.target.value)} disabled={loadingReferences || samples.length === 0}>{samples.map((sample) => <option key={sample.Sample_ID} value={sample.Sample_ID}>{sample.Sample_ID}</option>)}</Select></Field>
            {loadingReferences && <LoadingIndicator compact label="Loading requirements and samples…" />}
            {!loadingReferences && requirements.length === 0 && <p className="text-xs text-muted">No requirements are configured for this plant.</p>}
          </div>
          <div className="mt-5 border-t border-line pt-5">
            <Field label="Saved height session" hint="Sessions are loaded from the C3 height-measurement results directory.">
              <Select value={sessionId} onChange={(event) => { setSessionId(event.target.value); setResult(null); }} disabled={loadingSampleData || sessions.length === 0}>
                {sessions.length === 0 && <option value="">No saved sessions for this sample</option>}
                {sessions.map((session) => <option key={session.session_id} value={session.session_id}>{formatDate(session.timestamp)} · {session.leaf_count} leaves · {session.session_id}</option>)}
              </Select>
            </Field>
            {selectedSession && <div className="mt-3 flex flex-wrap gap-2"><StatusBadge tone={selectedSession.calibration_verified ? "ready" : "review"}>Calibration {selectedSession.calibration_verified ? "confirmed" : "unconfirmed"}</StatusBadge><StatusBadge tone={selectedSession.scale_reliable ? "ready" : "review"}>Scale {selectedSession.scale_reliable ? "confirmed" : "unreliable"}</StatusBadge></div>}
          </div>
          {loadingSampleData && <LoadingIndicator compact label="Loading images and sessions…" />}
          {sampleDataError && <div className="mt-4"><ErrorMessage title="Could not load sample data">{sampleDataError}</ErrorMessage></div>}
          <Button className="mt-5 w-full" disabled={evaluating || loadingSampleData || !sessionId || !requirementId} onClick={runReevaluation}><RefreshCw size={16} className={evaluating ? "animate-spin" : ""} />{evaluating ? "Reevaluating session…" : "Reevaluate height session"}</Button>
        </Card>

        <div className="space-y-5">
          <Card className="p-5 sm:p-6"><div className="mb-4 flex items-center gap-2"><Leaf size={17} className="text-leaf" /><h3 className="font-display font-extrabold text-forest">Sample image</h3></div>
            {images.length > 0 ? <>
              <Field label="Image view"><Select value={imageId} onChange={(event) => setImageId(event.target.value)}>{images.map((image) => <option key={image.Image_ID} value={image.Image_ID}>{image.View_Type} · {image.Image_ID}</option>)}</Select></Field>
              {selectedImage && <div className="mt-3 overflow-hidden rounded-xl border border-line bg-[#f7f9f6]"><img src={previewUrl} alt={`${selectedPlant?.Plant_Name || plantId} sample ${sampleId} ${selectedImage.View_Type} view`} className="max-h-[380px] min-h-[180px] w-full object-contain" /><p className="border-t border-line bg-white px-3 py-2 text-xs text-muted">{selectedImage.Relative_Path}</p></div>}
            </> : loadingSampleData ? <LoadingIndicator label="Loading sample images…" /> : <div className="rounded-xl border border-dashed border-line px-4 py-8 text-center"><ImageIcon size={20} className="mx-auto text-muted" /><p className="mt-2 text-sm font-bold text-muted">No images linked to this sample</p></div>}
          </Card>

          <Card className="p-5 sm:p-6"><div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><p className="text-[11px] font-extrabold uppercase tracking-[.15em] text-leaf">Height Assessment</p><h3 className="mt-1 font-display text-lg font-extrabold text-forest">Saved measurement result</h3><p className="mt-1 text-xs text-muted">Height rules are reported separately from overall plant quality.</p></div>{result && <StatusBadge tone={statusTone(result.status)}>{result.status}</StatusBadge>}</div>
            {!result ? <div className="rounded-xl border border-dashed border-line px-4 py-7 text-center"><p className="text-sm font-bold text-muted">No height assessment displayed</p><p className="mt-1 text-xs text-muted">Choose a saved session and requirement, then reevaluate.</p></div> : <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[
                ["Requirement", result.requirement_id], ["Leaves measured", `${result.valid_leaves}/${result.leaf_count}`], ["Leaves in range", result.leaves_in_range], ["In-range share", result.percentage_in_range == null ? "—" : `${result.percentage_in_range.toFixed(1)}%`],
              ].map(([label, value]) => <div key={label} className="rounded-xl bg-[#f7f9f6] p-3"><p className="text-[10px] font-bold uppercase tracking-wide text-muted">{label}</p><p className="mt-1 text-sm font-extrabold text-forest">{value}</p></div>)}</div>
              <div className="mt-4 overflow-x-auto rounded-xl border border-line"><table className="w-full min-w-[420px] text-left text-xs"><thead className="bg-[#f7f9f6] text-muted"><tr><th className="px-3 py-2.5 font-bold">Leaf</th><th className="px-3 py-2.5 font-bold">Height</th><th className="px-3 py-2.5 font-bold">Requirement</th></tr></thead><tbody>{result.leaves.map((leaf) => <tr key={leaf.leaf_number} className="border-t border-line"><td className="px-3 py-2.5 font-semibold text-ink">{leaf.leaf_number}</td><td className="px-3 py-2.5 text-ink">{typeof leaf.height_cm === "number" ? `${leaf.height_cm.toFixed(2)} cm` : "Unavailable"}</td><td className="px-3 py-2.5">{leaf.within_requirement == null ? <span className="text-muted">Not evaluated</span> : <StatusBadge tone={leaf.within_requirement ? "pass" : "fail"}>{leaf.within_requirement ? "In range" : "Out of range"}</StatusBadge>}</td></tr>)}</tbody></table></div>
              {result.reasons.length > 0 && <div className="mt-4 rounded-xl bg-amber-50 p-3"><p className="text-xs font-bold text-amber-900">Review reasons</p><ul className="mt-1 list-inside list-disc space-y-1 text-xs leading-5 text-amber-900/80">{result.reasons.map((reason, index) => <li key={`${reason}-${index}`}>{reason}</li>)}</ul></div>}
              {selectedRequirement && <p className="mt-3 text-[11px] text-muted">{result.buyer_id} · acceptable range {selectedRequirement.Min_Height_cm}–{selectedRequirement.Max_Height_cm} cm · session {result.session_id}</p>}
            </>}
          </Card>
        </div>
      </div>
    </>}
  </>;
}
