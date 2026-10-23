import test from "node:test";
import assert from "node:assert/strict";
import { validateFrontImage, validateInspectionImages } from "./uploadValidation.js";

test("upload accepts JPEG and PNG at the byte limit", () => {
  for (const type of ["image/jpeg", "image/png"]) {
    assert.equal(validateFrontImage({ type, size: 25 * 1024 * 1024 }), "");
  }
});

test("upload rejects missing, empty, oversized and unsupported files", () => {
  assert.match(validateFrontImage(null), /Choose/);
  assert.match(validateFrontImage({ type: "image/png", size: 0 }), /empty/);
  assert.match(validateFrontImage({ type: "image/jpeg", size: 25 * 1024 * 1024 + 1 }), /25 MiB/);
  assert.match(validateFrontImage({ type: "image/svg+xml", size: 100 }), /JPEG or PNG/);
});


test("new plant inspection requires Front and Root but permits optional Stem", () => {
  const file = { type: "image/png", size: 1024 };
  assert.match(validateInspectionImages({}), /Front image is required/);
  assert.match(validateInspectionImages({ Front: file }), /Root image is required/);
  assert.equal(validateInspectionImages({ Front: file, Root: file, Stem: null }), "");
  assert.equal(validateInspectionImages({ Front: file, Root: file, Stem: file }), "");
  assert.match(validateInspectionImages({ Front: file, Root: file, Stem: { type: "image/png", size: 0 } }), /Stem:.*empty/);
  assert.match(validateInspectionImages({ Front: file, Root: { type: "image/png", size: 26 * 1024 * 1024 } }), /Root:.*25 MiB/);
});
