import test from "node:test";
import assert from "node:assert/strict";
import { measurementAvailability, missingImageViews } from "./assessmentState.js";

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
