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

export function usesBranchDistribution(plantId) {
  return ["P02", "P03", "P04", "P05", "P06", "P07"].includes(plantId);
}

export function assessmentPresentation(plantId, result) {
  const definitions = [
    ["Root distribution", ["Root distribution"]],
    usesBranchDistribution(plantId)
      ? ["Branch distribution", ["Leaf distribution", "Branch distribution"]]
      : null,
    ["Height", ["Height"]],
    ["P01", "P08"].includes(plantId) ? ["Leaf count", ["Leaf count"]] : null,
    ["Overall visual quality", ["Overall visual quality", "Visual quality"]],
  ].filter(Boolean);
  const checks = result?.Checks || [];
  const visibleNames = definitions.flatMap(([, names]) => names);
  return {
    checks: definitions.map(([label, names]) => {
      const check = checks.find((item) => names.includes(item.Check));
      const status = check?.Presentation_Status || check?.Status;
      const presentationStatus = check?.Missing_Measurement || result?.Missing_Measurements?.some((name) => names.includes(name))
        ? "PENDING" : ["PASS", "FAIL", "PENDING", "REVIEW"].includes(status) ? status : "REVIEW";
      return {
        label,
        status: check ? presentationStatus : "PENDING",
        explanation: !check ? "Inspection not recorded."
          : presentationStatus === "PENDING" ? "Required inspection or measurement is missing."
          : presentationStatus === "REVIEW" ? "Recorded inspection needs review."
          : label === "Leaf count" || label === "Height" ? check.Reason : "",
      };
    }),
    additionalRequirements: ["FAIL", "MANUAL REVIEW", "REVIEW"].includes(result?.Buyer_Rule_Result) &&
      checks.some((check) => !visibleNames.includes(check.Check) &&
        (["FAIL", "MANUAL REVIEW", "REVIEW"].includes(check.Status) ||
          ["FAIL", "REVIEW", "PENDING"].includes(check.Presentation_Status))),
  };
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
