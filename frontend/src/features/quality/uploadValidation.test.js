import test from "node:test";
import assert from "node:assert/strict";
import { validateFrontImage } from "./uploadValidation.js";

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
