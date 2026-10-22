    import apiClient from "./apiClient.js";

export async function askAdvisor(question, variety) {
  const { data } = await apiClient.post("/chat", { question, variety }, { timeout: 180_000 });
  return data;
}