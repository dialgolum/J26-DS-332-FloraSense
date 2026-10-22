import apiClient from "./apiClient.js";

export async function identifyPlant(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await apiClient.post("/identify", form);
  return data;
}