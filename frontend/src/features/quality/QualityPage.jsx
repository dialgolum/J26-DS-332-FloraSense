import { useEffect, useRef, useState } from "react";
import { ClipboardCheck, Image as ImageIcon } from "lucide-react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import Field, { Select } from "../../components/ui/Field.jsx";
import Button from "../../components/ui/Button.jsx";
import StatusBadge from "../../components/ui/StatusBadge.jsx";
import LoadingIndicator from "../../components/ui/LoadingIndicator.jsx";
import ErrorMessage from "../../components/ui/ErrorMessage.jsx";
import {
  assessSample, getApiErrorMessage, getImagePreviewUrl, getPlantTypes,
  getRequirements, getSampleImages, getSamples,
  getHeightAnnotations,
} from "./api.js";

import { assessmentPresentation, canAssess, usesBranchDistribution } from "./assessmentState.js";

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
  const [heightAnnotations, setHeightAnnotations] = useState([]);
  const [loadingAnnotations, setLoadingAnnotations] = useState(false);
  const [annotationError, setAnnotationError] = useState("");
  const [loadingPlants, setLoadingPlants] = useState(true);
  const [loadingReferences, setLoadingReferences] = useState(false);
  const [loadingSampleData, setLoadingSampleData] = useState(false);
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
      setImages([]); setImageId(""); setLoadingSampleData(false);
      return undefined;
    }
    let active = true;
    setImages([]); setImageId("");
    setLoadingSampleData(true);
    setSampleDataError("");
    getSampleImages(sampleId, plantId).then((imageRows) => {
      if (!active) return;
      setImages(imageRows);
      setImageId(imageRows[0]?.Image_ID || "");
    }).catch((requestError) => {
      if (active) setSampleDataError(getApiErrorMessage(requestError, "Could not load plant photographs."));
    }).finally(() => { if (active) setLoadingSampleData(false); });
    return () => { active = false; };
  }, [plantId, sampleId]);

  useEffect(() => {
    let active = true;
    evaluationVersion.current += 1;
    setHeightAnnotations([]); setQualityResult(null); setAssessing(false);
    setAnnotationError(""); setLoadingAnnotations(false);
    if (!sampleId || !requirementId || loadingReferences) return undefined;
    setLoadingAnnotations(true);
    getHeightAnnotations(sampleId, requirementId)
      .then((heights) => {
        if (active) setHeightAnnotations(heights);
      }).catch((requestError) => {
        if (active) setAnnotationError(getApiErrorMessage(requestError, "Could not load recorded annotations."));
      }).finally(() => { if (active) setLoadingAnnotations(false); });
    return () => { active = false; };
  }, [plantId, sampleId, requirementId, loadingReferences]);

  const selectedImage = images.find((item) => item.Image_ID === imageId);
  const selectedRequirement = requirements.find((item) => item.Requirement_ID === requirementId);
  const selectedSample = samples.find((item) => item.Sample_ID === sampleId);
  const presentation = assessmentPresentation(plantId, qualityResult);
  const assessmentReady = canAssess({ plantId, sampleId, requirementId, samples, requirements,
    loading: loadingReferences || loadingPlants || assessing });

  function invalidateSelection() {
    evaluationVersion.current += 1;
    setQualityResult(null); setAssessing(false); setError("");
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

  const extraImages = images.filter((image) => !["Front", "Root"].includes(image.View_Type));

  return <>
    <PageIntro eyebrow="Plant quality control" title="Buyer-specific quality assessment" description="Inspect plant quality against buyer requirements." />
    {error && <div className="mb-5"><ErrorMessage title="Inspection request failed">{error}</ErrorMessage></div>}
    {loadingPlants ? <Card className="p-5"><LoadingIndicator label="Loading plant types..." /></Card> : plants.length === 0 ? <Card className="p-5"><p className="text-sm text-muted">No plant types are available.</p></Card> :
      <div className="mx-auto max-w-5xl space-y-5">
        <Card className="p-5 sm:p-6">
          <h2 className="mb-5 font-display text-lg font-extrabold text-forest">Assessment setup</h2>
          <div className="grid gap-4 md:grid-cols-3">
            <Field label="Plant type"><Select aria-label="Plant type" value={plantId} onChange={(event) => { invalidateSelection(); setImages([]); setRequirements([]); setSamples([]); setSampleId(""); setRequirementId(""); setPlantId(event.target.value); }}>{plants.map((plant) => <option key={plant.Plant_ID} value={plant.Plant_ID}>{plant.Plant_ID} - {plant.Plant_Name}</option>)}</Select></Field>
            <Field label="Buyer requirement"><Select aria-label="Buyer requirement" value={requirementId} onChange={(event) => { invalidateSelection(); setRequirementId(event.target.value); }} disabled={loadingReferences || !requirements.length}>{!requirements.length && <option value="">No buyer profiles available</option>}{requirements.map((profile) => <option key={profile.Requirement_ID} value={profile.Requirement_ID}>{profile.Buyer_ID} - {profile.Requirement_ID}</option>)}</Select></Field>
            <Field label="Sample"><Select aria-label="Sample" value={sampleId} onChange={(event) => { invalidateSelection(); setImages([]); setSampleId(event.target.value); }} disabled={loadingReferences || !samples.length}>{!samples.length && <option value="">No samples available</option>}{samples.map((sample) => <option key={sample.Sample_ID} value={sample.Sample_ID}>{sample.Sample_ID}</option>)}</Select></Field>
          </div>
          {loadingReferences && <div className="mt-3"><LoadingIndicator compact label="Loading samples and buyer profiles..." /></div>}
          <div className="mt-5 flex flex-col gap-4 border-t border-line pt-5 sm:flex-row sm:items-center sm:justify-between">
            <label className="flex items-center gap-2 text-sm font-bold"><input type="checkbox" checked={aiEnabled} onChange={(event) => { invalidateSelection(); setAiEnabled(event.target.checked); }} />Include AI-assisted inspection</label>
            <Button disabled={!assessmentReady} onClick={runQualityAssessment}><ClipboardCheck size={16} />{assessing ? "Assessing sample..." : "Run Quality Assessment"}</Button>
          </div>
          {assessing && <div className="mt-3" role="status"><LoadingIndicator compact label="Inspecting plant quality..." /></div>}
        </Card>

        <Card className="p-5 sm:p-6">
          <h2 className="mb-4 font-display text-lg font-extrabold text-forest">Plant photographs</h2>
          {sampleDataError && <ErrorMessage title="Photographs unavailable">{sampleDataError}</ErrorMessage>}
          {loadingSampleData ? <LoadingIndicator label="Loading photographs..." /> : <>
            <div className="grid gap-4 sm:grid-cols-2">{["Front", "Root"].map((view) => {
              const photo = images.find((image) => image.View_Type === view);
              return <figure key={view} className="overflow-hidden rounded-xl border border-line bg-[#f7f9f6]"><figcaption className="px-4 py-3 text-sm font-bold">{view}</figcaption>{photo ? <SamplePhoto key={photo.Image_ID} image={photo} sampleId={sampleId} /> : <div className="px-4 py-12 text-center text-sm text-muted"><ImageIcon size={20} className="mx-auto mb-2" />No {view.toLowerCase()} photograph available</div>}</figure>;
            })}</div>
            {!!extraImages.length && <details className="mt-4"><summary className="cursor-pointer text-sm font-bold text-forest">Other photographs</summary><div className="mt-3"><Field label="View"><Select aria-label="Other photograph view" value={extraImages.some((image) => image.Image_ID === imageId) ? imageId : ""} onChange={(event) => setImageId(event.target.value)}><option value="">Select a view</option>{extraImages.map((image, index) => <option key={image.Image_ID} value={image.Image_ID}>{image.View_Type}{extraImages.filter((row) => row.View_Type === image.View_Type).length > 1 ? ` (${index + 1})` : ""}</option>)}</Select></Field>{selectedImage && !["Front", "Root"].includes(selectedImage.View_Type) && <SamplePhoto key={selectedImage.Image_ID} image={selectedImage} sampleId={sampleId} />}</div></details>}
          </>}
        </Card>

        <Card className="p-5 sm:p-6">
          <h2 className="font-display text-lg font-extrabold text-forest">Quality assessment results</h2>
          <p className="mt-2 text-xs text-muted">Based on recorded expert observations and measurements.</p>
          {!qualityResult ? <p className="mt-3 text-sm text-muted">Select a sample and buyer requirement, then run the quality assessment.</p> : <div aria-live="polite">
            <div className="mt-4 flex items-center justify-between gap-3"><p className="text-sm font-bold">Overall buyer assessment result</p><StatusBadge tone={statusTone(qualityResult.Buyer_Rule_Result)}>{qualityResult.Buyer_Rule_Result === "MANUAL REVIEW" ? "REVIEW" : qualityResult.Buyer_Rule_Result}</StatusBadge></div>
            {presentation.additionalRequirements && <p className="mt-3 text-sm text-muted">Additional buyer requirements affect this result.</p>}
            <ul className="mt-4 divide-y divide-line">{presentation.checks.map((check) => <li key={check.label} className="py-4"><div className="flex items-center justify-between gap-3"><h3 className="text-sm font-bold">{check.label}</h3><StatusBadge tone={statusTone(check.status)}>{check.status}</StatusBadge></div>{check.explanation && <p className="mt-2 text-sm text-muted">{check.explanation}</p>}</li>)}</ul>
          </div>}
        </Card>

        <Card className="p-5 sm:p-6">
          <h2 className="font-display text-lg font-extrabold text-forest">AI-assisted inspection</h2>
          <p className="mt-2 text-xs text-muted">AI findings support inspection and do not change the quality decision.</p>
          {!qualityResult ? <p className="mt-3 text-sm text-muted">No AI findings available yet.</p> : <div className="mt-4 grid gap-4 sm:grid-cols-2">{(usesBranchDistribution(plantId) ? ["root_B", "leaf_B"] : ["root_B"]).map((key) => {
            const observation = qualityResult.AI_Observations?.[key];
            const prediction = observation?.prediction;
            const available = observation?.status === "success" && Boolean(prediction?.predicted_label);
            const stateLabel = { disabled: "Not requested", unavailable: "Inspection unavailable", error: "Inspection unavailable", unsupported: "Not supported for this plant type", not_applicable: "Not applicable" }[observation?.status] || "Unavailable";
            return <div key={key} className="rounded-xl border border-line p-4"><div className="flex flex-wrap items-center gap-2"><h3 className="text-sm font-bold">{key === "root_B" ? "Root distribution prediction" : "Branch distribution prediction"}</h3><StatusBadge tone="neutral">AI-assisted</StatusBadge></div><p className="mt-2 text-sm font-bold">{available ? prediction.predicted_label : stateLabel}</p>
              {available && ["agreement", "disagreement"].includes(observation.comparison_status) && <div className="mt-2"><StatusBadge tone={observation.comparison_status === "disagreement" ? "review" : "neutral"}>{observation.comparison_status === "agreement" ? "Agrees with expert" : "Disagrees with expert"}</StatusBadge></div>}
              {!available && ["unavailable", "error"].includes(observation?.status) && <p className="mt-2 text-xs text-muted">Continue with the recorded inspection checks or request manual review.</p>}
            </div>;
          })}</div>}

        </Card>

        <Card className="p-5 sm:p-6"><details><summary className="cursor-pointer font-display text-lg font-extrabold text-forest">Inspection details</summary>
          {loadingAnnotations ? <LoadingIndicator label="Loading inspection details..." /> : annotationError ? <div className="mt-4"><ErrorMessage title="Inspection details unavailable">{annotationError}</ErrorMessage></div> : <div className="mt-4 space-y-4">
            <StatusBadge tone="neutral">Recorded inspection</StatusBadge>
            <p className="text-sm">Root status: {selectedSample?.Observed_Root_Status ?? "Not recorded"}</p>
            {selectedRequirement && <p className="text-sm">Buyer {selectedRequirement.Buyer_ID} - Requirement {selectedRequirement.Requirement_ID}{selectedRequirement.Min_Height_cm != null && selectedRequirement.Max_Height_cm != null ? ` | Height range: ${selectedRequirement.Min_Height_cm}-${selectedRequirement.Max_Height_cm} cm` : ""}</p>}
            {!heightAnnotations.length && <p className="text-sm text-muted">No height measurement recorded for this buyer requirement.</p>}
            {heightAnnotations.map((row, index) => <div key={index} className="rounded-xl border border-line p-3 text-sm"><p className="font-bold">Recorded height assessment: {row.Height_Result ?? "Not recorded"}</p><p className="mt-1">Leaves in range: {row.Leaves_In_Range ?? "Not recorded"} / {row.Total_Leaves_Assessed ?? "Not recorded"} assessed</p>{row.Measured_Stem_Height != null && <p>Stem height: {row.Measured_Stem_Height} cm</p>}</div>)}

          </div>}
        </details></Card>
      </div>}
  </>;
}
