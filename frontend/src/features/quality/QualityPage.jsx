import { useEffect, useMemo, useRef, useState } from "react";
import { ClipboardCheck, Image as ImageIcon, Leaf, RefreshCw } from "lucide-react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import Field, { Select } from "../../components/ui/Field.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import LoadingIndicator from "../../components/ui/LoadingIndicator.jsx";
import ErrorMessage from "../../components/ui/ErrorMessage.jsx";
import {
  assessSample, getApiErrorMessage, getHeightSessions, getImagePreviewUrl, getPlantTypes,
  getRequirements, getSampleImages, getSamples, reevaluateHeightSession,
  getHeightAnnotations, getQualityAnnotations,
} from "./api.js";

import { canAssess, checkSummary, measurementAvailability, missingImageViews } from "./assessmentState.js";
import AutomaticLeafUpload from "./AutomaticLeafUpload.jsx";

function statusTone(status) {
  if (status === "PASS") return "pass";
  if (status === "FAIL") return "fail";
  if (status === "MANUAL REVIEW" || status === "REVIEW") return "review";
  if (status === "PENDING") return "pending";
  return "neutral";
}

function SamplePhoto({ image, sampleId }) {
  const [failed, setFailed] = useState(false);
  return failed ? <p role="status" className="grid h-40 place-items-center rounded-xl bg-slate-50 p-4 text-sm text-muted">{image.View_Type} photograph unavailable</p> :
    <img src={getImagePreviewUrl(image.Image_ID)} onError={() => setFailed(true)} alt={`${sampleId} ${image.View_Type} photograph`} className="max-h-[380px] min-h-[180px] w-full object-contain" />;
}

function formatDate(value) {
  if (!value) return "Date unavailable";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

export default function QualityPage() {
  const evaluationVersion = useRef(0);
  const [qualityResult, setQualityResult] = useState(null);
  const [assessing, setAssessing] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(true);
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
  const [heightAnnotations, setHeightAnnotations] = useState([]);
  const [qualityAnnotations, setQualityAnnotations] = useState([]);
  const [loadingAnnotations, setLoadingAnnotations] = useState(false);
  const [annotationError, setAnnotationError] = useState("");
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
    setRequirements([]); setRequirementId(""); setSamples([]); setSampleId("");
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
      setImages([]); setImageId(""); setSessions([]); setSessionId(""); setLoadingSampleData(false);
      return undefined;
    }
    let active = true;
    setImages([]); setImageId(""); setSessions([]); setSessionId("");
    setLoadingSampleData(true);
    setSampleDataError("");
    setResult(null);
    Promise.allSettled([getSampleImages(sampleId, plantId), getHeightSessions(sampleId, plantId)]).then(([imageRows, sessionRows]) => {
      if (!active) return;
      const failures = [imageRows, sessionRows].filter((item) => item.status === "rejected");
      if (failures.length) setSampleDataError(getApiErrorMessage(failures[0].reason, "Some sample data is unavailable."));
      imageRows = imageRows.status === "fulfilled" ? imageRows.value : [];
      sessionRows = sessionRows.status === "fulfilled" ? sessionRows.value : [];
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
  }, [plantId, sampleId]);

  useEffect(() => {
    let active = true;
    evaluationVersion.current += 1;
    setHeightAnnotations([]); setQualityAnnotations([]); setResult(null); setQualityResult(null); setAssessing(false);
    setAnnotationError(""); setLoadingAnnotations(false);
    if (!sampleId || !requirementId || loadingReferences) return undefined;
    setLoadingAnnotations(true);
    Promise.all([getHeightAnnotations(sampleId, requirementId), getQualityAnnotations(sampleId, requirementId)])
      .then(([heights, quality]) => {
        if (active) { setHeightAnnotations(heights); setQualityAnnotations(quality); }
      }).catch((requestError) => {
        if (active) setAnnotationError(getApiErrorMessage(requestError, "Could not load recorded annotations."));
      }).finally(() => { if (active) setLoadingAnnotations(false); });
    return () => { active = false; };
  }, [plantId, sampleId, requirementId, loadingReferences]);

  const selectedImage = images.find((item) => item.Image_ID === imageId);
  const selectedRequirement = requirements.find((item) => item.Requirement_ID === requirementId);
  const selectedSession = sessions.find((item) => item.session_id === sessionId);
  const selectedSample = samples.find((item) => item.Sample_ID === sampleId);
  const summary = useMemo(() => checkSummary(qualityResult?.Checks), [qualityResult]);
  const assessmentReady = canAssess({ plantId, sampleId, requirementId, samples, requirements,
    loading: loadingReferences || loadingPlants || assessing });

  function invalidateSelection() {
    evaluationVersion.current += 1;
    setQualityResult(null); setResult(null); setAssessing(false); setEvaluating(false); setError("");
  }

  async function runQualityAssessment() {
    if (!assessmentReady) return;
    const version = ++evaluationVersion.current;
    setAssessing(true); setQualityResult(null); setError("");
    try {
      const assessed = await assessSample(plantId, sampleId, requirementId, aiEnabled);
      if (version === evaluationVersion.current) setQualityResult(assessed);
    } catch (requestError) {
      if (version === evaluationVersion.current) setError(getApiErrorMessage(requestError, "Quality assessment failed. Please retry."));
    } finally {
      if (version === evaluationVersion.current) setAssessing(false);
    }
  }

  async function runReevaluation() {
    if (!sessionId || !requirementId) return;
    const version = ++evaluationVersion.current;
    setEvaluating(true);
    setError("");
    setResult(null);
    try {
      const evaluated = await reevaluateHeightSession(sessionId, requirementId);
      if (version === evaluationVersion.current) setResult(evaluated);
    } catch (requestError) {
      if (version === evaluationVersion.current) setError(getApiErrorMessage(requestError, "Could not reevaluate this height session."));
    } finally {
      if (version === evaluationVersion.current) setEvaluating(false);
    }
  }

  return <>
    <PageIntro eyebrow="C3 · Export readiness" title="Buyer-specific quality assessment" description="Assess annotated plant samples against buyer requirements, with optional advisory root and branch-origin observations." action={<StatusBadge tone="neutral">Research prototype</StatusBadge>} />
    {error && <div className="mb-5"><ErrorMessage title="C3 request failed">{error}</ErrorMessage></div>}
    {loadingPlants ? <Card className="p-5"><LoadingIndicator label="Loading plant types…" /></Card> : plants.length === 0 ? <Card className="p-5"><p className="text-sm text-muted">No plant types are available.</p></Card> : <>
      <div className="grid items-start gap-5 xl:grid-cols-[.85fr_1.15fr]">
        <Card className="p-5 sm:p-6">
          <div className="mb-5 flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-forest"><ClipboardCheck size={19} /></span><div><p className="font-display font-extrabold text-forest">Assessment setup</p><p className="text-xs text-muted">Options loaded from the C3 workbook</p></div></div>
          <div className="space-y-4">
            <Field label="Plant type"><Select value={plantId} aria-label="Plant type" onChange={(event) => { invalidateSelection(); setImages([]); setSessions([]); setRequirements([]); setSamples([]); setSampleId(""); setRequirementId(""); setPlantId(event.target.value); }}>{plants.map((plant) => <option key={plant.Plant_ID} value={plant.Plant_ID}>{plant.Plant_ID} · {plant.Plant_Name}</option>)}</Select></Field>
            <Field label="Buyer requirement"><Select value={requirementId} aria-label="Buyer requirement" onChange={(event) => { invalidateSelection(); setRequirementId(event.target.value); }} disabled={loadingReferences || requirements.length === 0}>{requirements.map((profile) => <option key={profile.Requirement_ID} value={profile.Requirement_ID}>{profile.Requirement_ID} · {profile.Buyer_ID} · {profile.Min_Height_cm}–{profile.Max_Height_cm} cm</option>)}</Select></Field>
            <Field label="Sample"><Select value={sampleId} aria-label="Sample" onChange={(event) => { invalidateSelection(); setImages([]); setSessions([]); setSampleId(event.target.value); }} disabled={loadingReferences || samples.length === 0}>{samples.map((sample) => <option key={sample.Sample_ID} value={sample.Sample_ID}>{sample.Sample_ID}</option>)}</Select></Field>
            {loadingReferences && <LoadingIndicator compact label="Loading requirements and samples…" />}
            {!loadingReferences && requirements.length === 0 && <p className="text-xs text-muted">No requirements are configured for this plant.</p>}
          </div>
          <div className="mt-5 border-t border-line pt-5">
            <label className="flex items-center gap-2 text-sm font-bold"><input type="checkbox" checked={aiEnabled} onChange={(event) => { invalidateSelection(); setAiEnabled(event.target.checked); }} />Include Root B and Leaf B advisory AI</label>
            <p className="mt-2 text-xs leading-5 text-muted">Uses server-configured checkpoints. Unavailable models are reported without blocking buyer rules.</p>
            <Button className="mt-4 w-full" disabled={!assessmentReady || evaluating} onClick={runQualityAssessment}><ClipboardCheck size={16} />{assessing ? "Assessing sample..." : "Run Quality Assessment"}</Button>
            {assessing && <div className="mt-3" role="status"><LoadingIndicator compact label="Running buyer rules and requested AI observations..." /></div>}
          </div>
          <div className="mt-5 border-t border-line pt-5">
            <Field label="Saved height session" hint="Sessions are loaded from the C3 height-measurement results directory.">
              <Select value={sessionId} onChange={(event) => { evaluationVersion.current += 1; setSessionId(event.target.value); setResult(null); }} disabled={assessing || evaluating || loadingSampleData || sessions.length === 0}>
                {sessions.length === 0 && <option value="">No saved sessions for this sample</option>}
                {sessions.map((session) => <option key={session.session_id} value={session.session_id}>{formatDate(session.timestamp)} · {session.leaf_count} leaves · {session.session_id}</option>)}
              </Select>
            </Field>
            {selectedSession && <div className="mt-3 flex flex-wrap gap-2"><StatusBadge tone={selectedSession.calibration_verified ? "ready" : "review"}>Calibration {selectedSession.calibration_verified ? "confirmed" : "unconfirmed"}</StatusBadge><StatusBadge tone={selectedSession.scale_reliable ? "ready" : "review"}>Scale {selectedSession.scale_reliable ? "confirmed" : "unreliable"}</StatusBadge></div>}
          </div>
          {loadingSampleData && <LoadingIndicator compact label="Loading images and sessions…" />}
          {sampleDataError && <div className="mt-4"><ErrorMessage title="Could not load sample data">{sampleDataError}</ErrorMessage></div>}
          <Button className="mt-5 w-full" disabled={assessing || evaluating || loadingSampleData || loadingReferences || !sessionId || !requirementId} onClick={runReevaluation}><RefreshCw size={16} className={evaluating ? "animate-spin" : ""} />{evaluating ? "Reevaluating session…" : "Reevaluate height session"}</Button>
        </Card>

        <div className="space-y-5">
          <Card className="p-5 sm:p-6">
            <h2 className="font-display text-lg font-extrabold text-forest">Quality assessment</h2>
            {!qualityResult ? <p className="mt-3 text-sm text-muted">Choose a sample and buyer profile, then run an assessment. Results use the recorded expert annotations.</p> : <div aria-live="polite">
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3"><p className="text-sm font-bold">Original buyer-rule decision</p><StatusBadge tone={statusTone(qualityResult.Buyer_Rule_Result)}>{qualityResult.Buyer_Rule_Result}</StatusBadge></div>
              <p className="mt-3 text-xs text-muted">Plant {qualityResult.Plant_ID} | Sample {qualityResult.Sample_ID} | Buyer {qualityResult.Buyer_ID} | Requirement {qualityResult.Requirement_ID}</p>
              <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">{Object.entries(summary).map(([status, count]) => <div key={status} className="rounded-xl border border-line p-3"><StatusBadge tone={statusTone(status)}>{status}</StatusBadge><p className="mt-2 text-xl font-bold">{count}</p></div>)}</div>
              {qualityResult.Missing_Measurements?.length > 0 && <p className="mt-4 rounded-xl bg-amber-50 p-3 text-sm text-amber-900">Required measurements pending: {qualityResult.Missing_Measurements.join(", ")}. Missing height counts require manual measurement; no height is inferred from these classifiers. Unresolved mandatory checks prevent approval.</p>}
              <ul className="mt-4 divide-y divide-line">{qualityResult.Checks.map((check, index) => <li key={`${check.Check}-${index}`} className="py-4"><div className="flex items-center justify-between gap-3"><h3 className="text-sm font-bold">{check.Check}</h3><StatusBadge tone={statusTone(check.Presentation_Status)}>{check.Presentation_Status}</StatusBadge></div><p className="mt-2 text-sm text-muted">{check.Reason}</p><p className="mt-1 text-xs text-muted">Original rule status: {check.Status}</p></li>)}</ul>
              {qualityResult.Manual_Review?.required && <details className="mt-3 rounded-xl bg-amber-50 p-3 text-sm text-amber-900"><summary className="cursor-pointer font-bold">Manual-review reasons</summary><ul className="mt-2 list-inside list-disc space-y-2">{qualityResult.Manual_Review.reasons.map((reason, index) => <li key={index}>{reason}</li>)}</ul></details>}
            </div>}
          </Card>
          <Card className="p-5 sm:p-6">
            <h2 className="font-display text-lg font-extrabold text-forest">Advisory AI observations</h2>
            <p className="mt-2 text-xs leading-5 text-muted">Model scores are uncalibrated softmax values, not reliable probabilities of correctness. AI never changes the buyer-rule decision or replaces expert annotations.</p>
            {!qualityResult ? <p className="mt-3 text-sm text-muted">No predictions requested yet.</p> : <div className="mt-4 grid gap-4 sm:grid-cols-2">{["root_B", "leaf_B"].map((key) => {
              const observation = qualityResult.AI_Observations[key];
              const prediction = observation?.prediction;
              return <div key={key} className="rounded-xl border border-line p-4"><h3 className="text-sm font-bold">{key === "root_B" ? "Root B | Root distribution" : "Leaf B | Branch-origin distribution"}</h3>
                <p className="mt-2 text-sm font-bold">{observation?.status === "success" ? prediction.predicted_label : observation?.status?.replaceAll("_", " ") || "Unavailable"}</p>
                {observation?.status === "success" && <><p className="mt-2 text-sm">Model score: {Number.isFinite(prediction.confidence) ? `${(prediction.confidence * 100).toFixed(1)}%` : "Unavailable"}</p><p className="mt-2 text-sm">Expert label: {observation.expert_annotation?.label || "Missing or unclear"}</p><StatusBadge tone={observation.comparison_status === "disagreement" ? "review" : "neutral"}>{observation.comparison_status === "agreement" ? "Agrees with expert" : observation.comparison_status === "disagreement" ? "Disagrees | expert review" : "Expert comparison unavailable"}</StatusBadge><details className="mt-3 text-xs text-muted"><summary className="cursor-pointer">Model metadata</summary><p className="mt-2 break-words">{prediction.architecture} | {prediction.model_variant} | {prediction.dataset_version}</p><p className="mt-1 break-all">{prediction.checkpoint_id}</p></details></>}
                <p className="mt-2 text-xs leading-5 text-muted">{observation?.reason}</p>
              </div>;
            })}</div>}
            <p className="mt-3 text-xs leading-5 text-muted">Leaf distribution describes qualifying branches originating on opposite sides of the main stem. Foliage extending across the stem or equal leaf counts do not establish Both Sides.</p>
          </Card>

          <Card className="p-5 sm:p-6"><div className="mb-4 flex items-center gap-2"><Leaf size={17} className="text-leaf" /><h3 className="font-display font-extrabold text-forest">Workbook-linked images</h3></div>
            {images.length > 0 ? <>
              <div className="mb-4 grid gap-3 sm:grid-cols-2">{["Front", "Root"].map((view) => {
                const photo = images.find((image) => image.View_Type === view);
                return <figure key={view} className="rounded-xl border border-line bg-[#f7f9f6] p-3"><figcaption className="mb-2 text-xs font-bold">{view} photograph</figcaption>{photo ? <SamplePhoto key={photo.Image_ID} image={photo} sampleId={sampleId} /> : <p className="py-10 text-center text-sm text-muted">No {view.toLowerCase()} photograph recorded</p>}</figure>;
              })}</div>
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">{images.map((image) => <button type="button" key={image.Image_ID} onClick={() => setImageId(image.Image_ID)} aria-pressed={imageId === image.Image_ID} className={`overflow-hidden rounded-xl border p-2 text-xs ${imageId === image.Image_ID ? "border-leaf bg-mint" : "border-line"}`}>
                <span className="block text-forest"><ImageIcon size={18} className="mx-auto" /></span>
                <span className="mt-2 block font-bold">{image.View_Type} | {image.Image_ID}</span>
              </button>)}</div>
              <p className="mb-3 text-xs text-muted">Views without workbook images: {missingImageViews(images).join(", ") || "None"}</p>
              <Field label="Image view"><Select value={imageId} onChange={(event) => setImageId(event.target.value)}>{images.map((image) => <option key={image.Image_ID} value={image.Image_ID}>{image.View_Type} · {image.Image_ID}</option>)}</Select></Field>
              {selectedImage && !["Front", "Root"].includes(selectedImage.View_Type) && <div className="mt-3 overflow-hidden rounded-xl border border-line bg-[#f7f9f6]"><SamplePhoto key={selectedImage.Image_ID} image={selectedImage} sampleId={sampleId} /><p className="border-t border-line bg-white px-3 py-2 text-xs text-muted">{selectedImage.View_Type} | {selectedImage.Image_ID}</p></div>}
            </> : loadingSampleData ? <LoadingIndicator label="Loading sample images…" /> : <div className="rounded-xl border border-dashed border-line px-4 py-8 text-center"><ImageIcon size={20} className="mx-auto text-muted" /><p className="mt-2 text-sm font-bold text-muted">No images linked to this sample</p></div>}
          </Card>

          <Card className="p-5 sm:p-6">
            <h3 className="font-display text-lg font-extrabold text-forest">Recorded manual annotations</h3>
            <p className="mt-1 text-xs text-muted">Source: Excel workbook. Quality checks apply to the sample; recorded height compliance applies to the selected buyer requirement.</p>
            {loadingAnnotations ? <LoadingIndicator label="Loading manual annotations..." /> : annotationError ? <ErrorMessage title="Annotations unavailable">{annotationError}</ErrorMessage> : <>
              <StatusBadge tone="neutral">Manual Annotation</StatusBadge>
              {heightAnnotations.length === 0 && <p className="mt-3 text-sm text-muted">No recorded height annotation for this requirement.</p>}
              {heightAnnotations.map((row, index) => <div key={index} className="mt-3 rounded-xl border border-line p-3 text-sm">
                <p className="font-bold">Recorded height compliance | {row.Requirement_ID}</p>
                <p>Height result: {row.Height_Result ?? "Not recorded"}</p>
                <p>Leaves in range: {row.Leaves_In_Range ?? "Not recorded"} / {row.Total_Leaves_Assessed ?? "Not recorded"} assessed</p>
              </div>)}
              {qualityAnnotations.length === 0 && <p className="mt-3 text-sm text-muted">No recorded quality annotations.</p>}
              {qualityAnnotations.map((row, index) => <dl key={index} className="mt-3 grid grid-cols-2 gap-3 text-sm">{[
                ["Leaf count", row.Total_Leaf_Count], ["Leaf distribution", row.Leaf_Distribution],
                ["Root distribution", row.Root_Distribution], ["Oasis damaged", row.Oasis_Damaged],
                ["General visual quality", row.General_visual_Quality],
              ].map(([label, value]) => <div key={label} className="rounded-xl bg-[#f7f9f6] p-3"><dt className="text-xs text-muted">{label}</dt><dd className="font-bold">{value ?? "Not recorded"}</dd></div>)}</dl>)}
              <p className="mt-3 text-sm">Recorded root condition: {selectedSample?.Observed_Root_Status ?? "Not recorded"}</p>
              <p className="mt-1 text-sm">Recorded packing mode: {selectedSample?.Observed_Packing_Mode ?? "Not recorded"}</p>
            </>}
          </Card>

          <Card className="p-5 sm:p-6"><div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><p className="text-[11px] font-extrabold uppercase tracking-[.15em] text-leaf">Height Assessment</p><h3 className="mt-1 font-display text-lg font-extrabold text-forest">Image-based CV height measurements</h3><p className="mt-1 text-xs text-muted">Saved image measurements with buyer-specific height rule-engine decisions. This is not an overall quality decision.</p></div>{result && <StatusBadge tone={statusTone(result.status)}>{result.status}</StatusBadge>}</div>
            {!result ? <div className="rounded-xl border border-dashed border-line px-4 py-7 text-center"><p className="text-sm font-bold text-muted">{measurementAvailability({ loading: loadingSampleData || loadingAnnotations, error: sampleDataError || annotationError, sessions, heightAnnotations })}</p><p className="mt-1 text-xs text-muted">{sessions.length ? "Choose a saved session and requirement, then reevaluate." : "Image-based reevaluation requires a saved height session."}</p></div> : <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[
                ["Buyer height decision", result.status], ["Source image", result.image_id ?? "Unavailable"], ["Requirement", result.requirement_id], ["Leaves measured", `${result.valid_leaves}/${result.leaf_count}`], ["Leaves in range", result.leaves_in_range], ["In-range share", result.percentage_in_range == null ? "—" : `${result.percentage_in_range.toFixed(1)}%`],
              ].map(([label, value]) => <div key={label} className="rounded-xl bg-[#f7f9f6] p-3"><p className="text-[10px] font-bold uppercase tracking-wide text-muted">{label}</p><p className="mt-1 text-sm font-extrabold text-forest">{value}</p></div>)}</div>
              <div className="mt-4 overflow-x-auto rounded-xl border border-line"><table className="w-full min-w-[420px] text-left text-xs"><thead className="bg-[#f7f9f6] text-muted"><tr><th className="px-3 py-2.5 font-bold">Leaf</th><th className="px-3 py-2.5 font-bold">Height</th><th className="px-3 py-2.5 font-bold">Requirement</th></tr></thead><tbody>{result.leaves.map((leaf) => <tr key={leaf.leaf_number} className="border-t border-line"><td className="px-3 py-2.5 font-semibold text-ink">{leaf.leaf_number}</td><td className="px-3 py-2.5 text-ink">{typeof leaf.height_cm === "number" ? `${leaf.height_cm.toFixed(2)} cm` : "Unavailable"}</td><td className="px-3 py-2.5">{leaf.within_requirement == null ? <span className="text-muted">Not evaluated</span> : <StatusBadge tone={leaf.within_requirement ? "pass" : "fail"}>{leaf.within_requirement ? "In range" : "Out of range"}</StatusBadge>}</td></tr>)}</tbody></table></div>
              {result.reasons.length > 0 && <div className="mt-4 rounded-xl bg-amber-50 p-3"><p className="text-xs font-bold text-amber-900">Review reasons</p><ul className="mt-1 list-inside list-disc space-y-1 text-xs leading-5 text-amber-900/80">{result.reasons.map((reason, index) => <li key={`${reason}-${index}`}>{reason}</li>)}</ul></div>}
              {selectedRequirement && <p className="mt-3 text-[11px] text-muted">{result.buyer_id} · acceptable range {selectedRequirement.Min_Height_cm}–{selectedRequirement.Max_Height_cm} cm · session {result.session_id}</p>}
            </>}
          </Card>
        </div>
      </div>
    </>}
    <details className="mt-6"><summary className="cursor-pointer text-sm font-bold text-muted">Experimental image upload tools</summary><div className="mt-4"><AutomaticLeafUpload /></div></details>
  </>;
}
