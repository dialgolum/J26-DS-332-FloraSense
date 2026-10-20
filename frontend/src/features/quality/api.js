import apiClient from "../../api/apiClient.js";

const PREFIX = "/api/v1/c3";

async function readData(request) {
  const response = await request;
  return response.data.data;
}

export const getPlantTypes = () => readData(apiClient.get(`${PREFIX}/plants`));
export const getRequirements = (plantId) => readData(apiClient.get(`${PREFIX}/requirements`, { params: { plant_id: plantId } }));
export const getSamples = (plantId) => readData(apiClient.get(`${PREFIX}/samples`, { params: { plant_id: plantId } }));
export const getSampleImages = (sampleId) => readData(apiClient.get(`${PREFIX}/samples/${encodeURIComponent(sampleId)}/images`));
export const getHeightSessions = (sampleId) => readData(apiClient.get(`${PREFIX}/samples/${encodeURIComponent(sampleId)}/height-sessions`));
export const reevaluateHeightSession = (sessionId, requirementId) => readData(apiClient.post(
  `${PREFIX}/height-sessions/${encodeURIComponent(sessionId)}/reevaluate`,
  { requirement_id: requirementId },
));

export const getImagePreviewUrl = (imageId) => apiClient.getUri({
  url: `${PREFIX}/images/${encodeURIComponent(imageId)}/preview`,
});

export function getApiErrorMessage(error, fallback) {
  return error?.response?.data?.error?.message || error?.message || fallback;
}
