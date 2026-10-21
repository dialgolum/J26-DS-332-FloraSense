export function measurementAvailability({ loading, error, sessions, heightAnnotations }) {
  if (loading) return "Loading measurement availability…";
  if (error) return "Measurement availability unavailable";
  if (sessions.length) return "Saved image-based measurements available";
  if (heightAnnotations.length) return "Manual Annotation";
  return "Measurement Required";
}

export function missingImageViews(images) {
  return ["Front", "Root", "Side", "Focus"].filter(
    (view) => !images.some((image) => image.View_Type === view),
  );
}

export function checkSummary(checks = []) {
  const counts = { PASS: 0, FAIL: 0, PENDING: 0, REVIEW: 0 };
  for (const check of checks) {
    const status = Object.hasOwn(counts, check.Presentation_Status) ? check.Presentation_Status : "REVIEW";
    counts[status] += 1;
  }
  return counts;
}

export function canAssess({ plantId, sampleId, requirementId, samples, requirements, loading }) {
  return !loading && Boolean(plantId && sampleId && requirementId) &&
    samples.some((row) => row.Plant_ID === plantId && row.Sample_ID === sampleId) &&
    requirements.some((row) => row.Plant_ID === plantId && row.Requirement_ID === requirementId);
}

export function inspectionReviewReasons(result) {
  if (!result) return [];
  const reasons = (result.Checks || [])
    .filter((check) => check.Status === "MANUAL REVIEW")
    .map((check) => `${check.Check}: ${check.Reason}`);
  for (const difference of result.Disagreements || []) {
    const feature = difference.task === "root" ? "Root condition" : "Branch distribution";
    reasons.push(`${feature}: AI finding ${difference.ai_label} differs from expert label ${difference.expert_label}. Review the plant.`);
  }
  for (const observation of Object.values(result.AI_Observations || {})) {
    if (["error", "unavailable"].includes(observation.status)) {
      reasons.push(`${observation.task === "root" ? "Root condition" : "Branch distribution"}: AI-assisted inspection is unavailable. Request manual inspection if needed.`);
    }
  }
  return [...new Set(reasons)];
}
