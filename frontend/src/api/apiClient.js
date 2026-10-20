import axios from "axios";

// The client is configured centrally. Feature pages do not call it until their
// corresponding FastAPI contract is implemented and enabled.
const apiClient = axios.create({
  baseURL: (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, ""),
  timeout: 30_000,
  headers: { Accept: "application/json" },
});

export default apiClient;
