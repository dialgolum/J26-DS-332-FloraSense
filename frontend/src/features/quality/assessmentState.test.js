import test from "node:test";
import assert from "node:assert/strict";
import { canAssess, checkSummary, inspectionReviewReasons, measurementAvailability, missingImageViews } from "./assessmentState.js";
import { assessmentPresentation, usesBranchDistribution } from "./assessmentState.js";

test("plant-specific presentation shows four checks and keeps missing checks pending", () => {
  for (const plant of ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]) {
    const branch = !["P01", "P08"].includes(plant);
    assert.equal(usesBranchDistribution(plant), branch);
    const rows = assessmentPresentation(plant, {}).checks;
    assert.deepEqual(rows.map((row) => row.label), branch
      ? ["Root distribution", "Branch distribution", "Height", "Overall visual quality"]
      : ["Root distribution", "Height", "Leaf count", "Overall visual quality"]);
    assert.ok(rows.every((row) => row.status === "PENDING"));
  }
});

test("presentation aliases existing fields without changing rule results", () => {
  const result = { Buyer_Rule_Result: "FAIL", Checks: [
    { Check: "Leaf distribution", Status: "PASS", Presentation_Status: "PASS" },
    { Check: "Packing mode", Status: "FAIL", Presentation_Status: "FAIL" },
    { Check: "Height", Status: "MANUAL REVIEW", Presentation_Status: "PENDING", Missing_Measurement: true },
    { Check: "Visual quality", Status: "MANUAL REVIEW" },
  ] };
  const original = JSON.stringify(result);
  const view = assessmentPresentation("P02", result);
  assert.equal(view.checks[1].status, "PASS");
  assert.equal(view.checks[2].status, "PENDING");
  assert.equal(view.checks[3].status, "REVIEW");
  assert.equal(view.additionalRequirements, true);
  assert.equal(JSON.stringify(result), original);
  result.Buyer_Rule_Result = "MANUAL REVIEW";
  result.Checks[1].Status = "MANUAL REVIEW";
  assert.equal(assessmentPresentation("P02", result).additionalRequirements, true);
  result.Checks.splice(1, 1);
  assert.equal(assessmentPresentation("P02", result).additionalRequirements, false);
});

test("saved CV sessions remain usable alongside manual annotations", () => {
  assert.equal(measurementAvailability({ sessions: [{}], heightAnnotations: [{}] }), "Saved image-based measurements available");
});

test("annotations provide a manual fallback without a saved session", () => {
  assert.equal(measurementAvailability({ sessions: [], heightAnnotations: [{}] }), "Manual Annotation");
});

test("missing data requires measurement, while errors and loading are distinct", () => {
  const empty = { sessions: [], heightAnnotations: [] };
  assert.equal(measurementAvailability(empty), "Measurement Required");
  assert.equal(measurementAvailability({ ...empty, error: "offline" }), "Measurement availability unavailable");
  assert.equal(measurementAvailability({ ...empty, loading: true }), "Loading measurement availability…");
});

test("multiple images and duplicate views do not invent unavailable views", () => {
  assert.deepEqual(missingImageViews([{ View_Type: "Front" }, { View_Type: "Root" }, { View_Type: "Front" }]), ["Side", "Focus"]);
  assert.deepEqual(missingImageViews(["Front", "Root", "Side", "Focus"].map((View_Type) => ({ View_Type }))), []);
  assert.deepEqual(missingImageViews([]), ["Front", "Root", "Side", "Focus"]);
});


test("presentation summary keeps unresolved checks separate from passed checks", () => {
  assert.deepEqual(checkSummary([
    { Status: "MANUAL REVIEW", Presentation_Status: "PENDING" },
    { Status: "MANUAL REVIEW", Presentation_Status: "REVIEW" },
    { Status: "PASS", Presentation_Status: "PASS" },
    { Status: "FAIL", Presentation_Status: "FAIL" },
  ]), { PASS: 1, FAIL: 1, PENDING: 1, REVIEW: 1 });
  assert.deepEqual(checkSummary(), { PASS: 0, FAIL: 0, PENDING: 0, REVIEW: 0 });
  assert.equal(checkSummary([{ Status: "PASS" }]).REVIEW, 1);
});

test("assessment requires compatible loaded plant, sample and buyer records", () => {
  const selection = { plantId: "P02", sampleId: "S001", requirementId: "R003",
    samples: [{ Plant_ID: "P02", Sample_ID: "S001" }],
    requirements: [{ Plant_ID: "P02", Requirement_ID: "R003" }], loading: false };
  assert.equal(canAssess(selection), true);
  assert.equal(canAssess({ ...selection, loading: true }), false);
  assert.equal(canAssess({ ...selection, sampleId: "" }), false);
  assert.equal(canAssess({ ...selection, requirements: [] }), false);
  assert.equal(canAssess({ ...selection, plantId: "P03" }), false);
  assert.equal(canAssess({ ...selection, samples: [{ Plant_ID: "P03", Sample_ID: "S001" }] }), false);
});


test("inspection review keeps rule reasons and disagreements without generic AI disclaimers", () => {
  const result = {
    Buyer_Rule_Result: "MANUAL REVIEW",
    Checks: [{ Check: "Height", Status: "MANUAL REVIEW", Reason: "Missing measurement." },
             { Check: "Packing", Status: "PASS", Reason: "Matches." }],
    Disagreements: [{ task: "leaf", expert_label: "Both Sides", ai_label: "One Side" }],
    AI_Observations: { root_B: { task: "root", status: "success" } },
    Manual_Review: { reasons: ["AI observation requires human review; score is uncalibrated."] },
  };
  const original = JSON.stringify(result);
  const reasons = inspectionReviewReasons(result);
  assert.equal(reasons.length, 2);
  assert.match(reasons[0], /Missing measurement/);
  assert.match(reasons[1], /One Side.*Both Sides/);
  assert.equal(JSON.stringify(result), original);
  assert.deepEqual(inspectionReviewReasons(null), []);
  assert.match(inspectionReviewReasons({ AI_Observations: { root_B: { task: "root", status: "error" } } })[0], /unavailable/);
});
