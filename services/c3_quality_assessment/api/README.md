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
