# FloraSense frontend foundation

React + Vite application using JavaScript/JSX, Tailwind CSS, React Router, Axios, and Lucide React. C3 Quality Assessment is connected to the C3 FastAPI reference-data, image-preview, and saved height-session reevaluation endpoints. C1, C2, and C4 remain UI-only; they do not send requests or generate model results.

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Vite prints the local URL (normally `http://localhost:5173`). To create and preview a production build:

```powershell
npm run build
npm run preview
```

Copy `.env.example` to `.env.local` to set `VITE_API_BASE_URL`. The default is `http://localhost:8000`. The current repository does not provide a connected shared backend.

## Structure

```text
src/
  app/                  Router and application composition
  api/                  Central Axios client and cross-feature API setup
  components/layout/    Shared shell, sidebar, and header
  components/ui/        Reusable buttons, cards, upload, notices, fields, badges
  features/<feature>/   Feature page and its API contract seam
  styles.css            Tailwind layers and shared visual primitives
```

Feature folders are independent ownership areas. Add page-specific UI and API request functions within the relevant `features/<name>/` folder; reuse common controls from `components/ui/`. Keep cross-feature transport settings in `src/api/apiClient.js`. Do not display predictions until the corresponding service returns them.

## Proposed API convention

Use a shared base URL from `VITE_API_BASE_URL` and version every route under `/api/v1`, for example `POST /api/v1/c1/identify` or `POST /api/v1/c3/assess`. Agree on request/response schemas, image upload limits, and error formats before wiring each feature. C3 calls the implemented `/api/v1/c3` endpoints. The other `features/*/api.js` files remain contract notes; their requests are not enabled.

## Current page state

- C1 identification and chatbot: interface ready; model/chat APIs pending.
- C2 disease detection: interface ready; model and response contract pending.
- C3 quality assessment: plant, requirement, sample, and image selectors use live workbook-backed API data. Image previews and saved P01 session reevaluation are connected. These are height-assessment results and are separate from overall quality decisions.
- C4 forecasting: chart shell ready; variables, horizon, model, and response contract pending. No forecast values are fabricated.
