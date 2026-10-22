# C3 FastAPI API

Run from the repository root after installing `requirements/base.txt` and the C3 requirements:

```powershell
python -m uvicorn services.c3_quality_assessment.api.main:app --reload --host 127.0.0.1 --port 8000
```

The app reads the existing C3 workbook through `data_loader.load_dataset()` and reads saved measurement sessions from `data/quality_assessment/results/height_measurements/`. It does not start the OpenCV measurement window or write to the workbook.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/v1/c3/plants` | List configured plant types |
| GET | `/api/v1/c3/requirements?plant_id=P01` | List buyer profiles for one plant |
| GET | `/api/v1/c3/samples?plant_id=P01` | List samples for one plant |
| GET | `/api/v1/c3/samples/{sample_id}/images` | List sample image metadata and preview paths |
| GET | `/api/v1/c3/images/{image_id}/preview` | Preview a workbook-linked image under the approved `data/quality_assessment/original/` tree |
| GET | `/api/v1/c3/samples/{sample_id}/height-sessions` | List valid saved sessions associated with a sample |
| POST | `/api/v1/c3/height-sessions/{session_id}/reevaluate` | Reuse a saved session against a requirement; JSON body: `{"requirement_id":"R001"}` |

Success responses use `{ "data": ..., "meta": ... }`. Errors use `{ "data": null, "error": { "code": ..., "message": ... } }`. CORS allows the Vite localhost origins by default; override them with comma-separated `FLORASENSE_CORS_ORIGINS` values.

Run focused tests with:

```powershell
python -m pytest services/c3_quality_assessment/tests/test_api.py
```

## Workbook annotations and browser verification

New read-only endpoints (both require `requirement_id=R001` or `R002`):

- `GET /api/v1/c3/samples/{sample_id}/height-assessments`: returns matching `Height_Assessments` rows for the sample and requirement.
- `GET /api/v1/c3/samples/{sample_id}/quality-annotations`: returns sample-wide `Quality_Annotations` rows. The requirement is validated against the sample's plant; quality values do not change with buyer. Metadata includes recorded sample root/packing observations.

Both identify their source as `recorded_manual_annotation`. Missing records return an empty list, missing cells return null, and unknown samples/requirements return 404. Incompatible plant requirements return 422. No annotation values are calculated or overwritten.

The Quality Assessment page lists all 19 P01 samples and every workbook-linked image, including repeated views. Thumbnails and the image selector open Front, Root, Side, or Focus images where linked. Missing views are listed explicitly. Manual quality checks and buyer-specific recorded height annotations appear separately from saved image-based CV measurements. Blank Leaf_Distribution cells show `Not recorded`. Root condition comes from `Samples.Observed_Root_Status`. AI visual-quality predictions have a separate panel indicating that no prediction was generated. Reevaluation produces a buyer-specific height rule decision, not an overall quality decision.

Start the API from the repository root:

```powershell
.venv/Scripts/python.exe -m uvicorn services.c3_quality_assessment.api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173/c3/quality` (or the localhost port Vite prints). Select plant P01, then:

| Sample | Views to open | Manual annotation with R001 | Manual annotation with R002 | Saved-session behavior |
|---|---|---|---|---|
| S004 | Front, Root; Side and Focus unavailable | 8 leaves; height PASS, 6/8 in range | height FAIL, 0/8 | No session: Manual Annotation; reevaluation disabled |
| S005 | Focus, Front, Root; Side unavailable | 11 leaves; height PASS, 7/11 in range | height FAIL, 2/11 | Select saved S005 session and click Reevaluate height session for each buyer |
| S014 | Front, Root; Side and Focus unavailable | 10 leaves; height PASS, 7/10 in range | height FAIL, 0/10 | No session: Manual Annotation; reevaluation disabled |

For each sample, check recorded root condition Rooted, root distribution Acceptable, oasis damage No, and general visual quality Acceptable. Leaf distribution is Not recorded. Switch R001/R002 and confirm sample-wide quality values remain unchanged.

For S005, R001 CV reevaluation must show 9 measured leaves, 5 in range (55.6%), PASS. Switch to R002 and reevaluate: 2/9 (22.2%), FAIL. The manual count remains 11 in its separate panel. The result identifies the saved session and source image, regardless of the image currently previewed. Changing sample, requirement, or session clears the displayed CV result.

If neither a saved session nor a matching height annotation exists, the CV panel displays `Measurement Required`. Current workbook samples all have recorded height annotations; this absence case is covered with isolated test data without editing the workbook.

Run validation:

```powershell
.venv/Scripts/python.exe -m pytest services/c3_quality_assessment/tests -q -p no:cacheprovider
node --test frontend/src/features/quality/assessmentState.test.js
cd frontend
npm run build
```

## Phase 1 automatic P01 front-image prototype

`POST /api/v1/c3/automatic/front-image?grid_profile_confirmed=true`

Send **raw JPEG or PNG bytes**, with the corresponding `Content-Type` (not multipart). `grid_profile_confirmed` defaults to false for new uploads. Confirm it only when this image uses the user-verified P01 graph paper: **1 cm major squares subdivided into five 0.2 cm intervals per axis**. The React upload section has a separate confirmation checkbox and resets it when a new file is chosen.

Validation rejects MIME/decoded-format mismatches, corrupt or truncated files, animations, empty uploads, files above 25 MiB, dimensions below 128 pixels or above 10000 pixels on either side, and images above 60 megapixels. The byte limit is enforced on both Content-Length and actual streamed bytes. At most two uploads are inspected concurrently per API process. No supplied filename is used as a path. Uploads and previews remain in memory, with no workbook, image or height-session writes.

The response envelope remains `{data, meta}`. Data contains `source: uploaded_image_opencv_prototype`, `estimated_count` (null when no usable estimate), `detected_region_count`, `leaf_regions`, `annotated_preview` (JPEG data URL), `status`, `count_reliable`, `reasons`, `diagnostics`, `calibration`, and coordinate dimensions. Region fields include a unique ID, processing-image bounding box and area, measured solidity and heuristic ambiguity flags. Statistical `confidence` is null rather than a fabricated percentage. EXIF orientation is applied and processing is bounded to a 1400-pixel longest edge. Native/oriented and processing dimensions are returned for coordinate conversion.

The untrained baseline selects green/yellow vegetation and uses distance peaks plus watershed to partition candidate regions. It retains small/partial candidates above a documented noise floor and warns about connected vegetation, possible overlap/duplicate splits, irregular contours, borders and threshold sensitivity. More than 256 seeds triggers an abstention rather than a truncated count. **Partitioning pixels does not establish one region per leaf. All automatic counts remain MANUAL REVIEW until validated.** No manual count enters inference, and the existing ResNet18 visual-quality classifier is not used for leaf detection.

Calibration attempts dominant background-line deskewing, then detects a five-to-one major/minor hierarchy independently in both axes and multiple background tiles. Equally spaced lines with unknown major/minor identity are rejected. Accepted evidence yields `grid_scale_candidate` and background-plane `scale_px_per_cm` in processing pixels; weak, ambiguous or perspective-inconsistent evidence yields `calibration_unreliable` and null physical scale. This does not verify plant coplanarity. Perspective rectification is deferred; scale is withheld when a single scale is unsupported.

`leaf_heights` is always empty because reliable individual emergence/tip endpoints have not been implemented in Phase 1. `quality_decision` is null. No overall plant height is substituted, no buyer rules run on tentative counts, and no root-distribution, root-health or disease model is invoked. The historical annotations and saved-session UI/API remain independent.

From the repository root, start the backend in PowerShell:

```powershell
Set-Location E:\ResearchProject\J26-DS-332-FloraSense
.\.venv\Scripts\python.exe -m uvicorn services.c3_quality_assessment.api.main:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal:

```powershell
Set-Location E:\ResearchProject\J26-DS-332-FloraSense\frontend
npm run dev -- --port 5173
```

Open `http://localhost:5173/c3/quality`. In **New P01 image**, choose a front image (for example `data/quality_assessment/original/individual/P01/P01_S005_front.jpg`), confirm the paper profile only if applicable, then click **Detect leaf regions**. Inspect the candidate count, orange numbered preview, MANUAL REVIEW reasons and calibration status. Use the separate historical section to check that S005 saved-session reevaluation and recorded manual count 11 still work. The uploaded result does not depend on the selected historical sample or buyer.

To call the upload endpoint directly:

```powershell
Set-Location E:\ResearchProject\J26-DS-332-FloraSense
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/c3/automatic/front-image?grid_profile_confirmed=true' -ContentType 'image/jpeg' -InFile '.\data\quality_assessment\original\individual\P01\P01_S005_front.jpg'
```

Regenerate the evaluation and run checks:

```powershell
.\.venv\Scripts\python.exe -m services.c3_quality_assessment.evaluate_automatic_leaf
.\.venv\Scripts\python.exe -m pytest services/c3_quality_assessment/tests -q -p no:cacheprovider
node --test frontend/src/features/quality/assessmentState.test.js frontend/src/features/quality/uploadValidation.test.js
Set-Location frontend
npm run build
```

See `docs/quality_assessment/phase1_leaf_count_evaluation.md`, the adjacent per-sample CSV/summary JSON, and `p01_image_inventory.json` for actual results and all 36 workbook-linked image dimensions. Evaluation previews/diagnostics are generated under the existing ignored `data/quality_assessment/processed/p01_leaf_prototype/` directory. Counts use sample-wide manual references; visible-front-leaf labels still need human annotation. The known S007 Front-to-S006-root linkage is flagged without modifying source data. No trained model is evaluated, and any future trained-model split must keep all views/crops of a physical plant together.
