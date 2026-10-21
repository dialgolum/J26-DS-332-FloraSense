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
