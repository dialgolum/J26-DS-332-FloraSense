import test from "node:test";
import assert from "node:assert/strict";
import { canAssess, checkSummary, measurementAvailability, missingImageViews } from "./assessmentState.js";

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
