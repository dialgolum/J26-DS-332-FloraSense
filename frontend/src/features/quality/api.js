import apiClient from "../../api/apiClient.js";

const PREFIX = "/api/v1/c3";

async function readData(request) {
  const response = await request;
  return response.data.data;
}

export const getPlantTypes = () => readData(apiClient.get(`${PREFIX}/plants`));
export const getRequirements = (plantId) => readData(apiClient.get(`${PREFIX}/requirements`, { params: { plant_id: plantId } }));
export const getSamples = (plantId) => readData(apiClient.get(`${PREFIX}/samples`, { params: { plant_id: plantId } }));
export const getSampleImages = (sampleId, plantId) => readData(apiClient.get(`${PREFIX}/samples/${encodeURIComponent(sampleId)}/images`, { params: { plant_id: plantId } }));
export const getHeightSessions = (sampleId, plantId) => readData(apiClient.get(`${PREFIX}/samples/${encodeURIComponent(sampleId)}/height-sessions`, { params: { plant_id: plantId } }));
export const inspectFrontImage = (file, gridProfileConfirmed, signal) => readData(apiClient.post(
  `${PREFIX}/automatic/front-image`, file, {
    headers: { "Content-Type": file.type }, signal,
    params: { grid_profile_confirmed: gridProfileConfirmed },
  },
));
export const getHeightAnnotations = (sampleId, requirementId) => readData(apiClient.get(
  `${PREFIX}/samples/${encodeURIComponent(sampleId)}/height-assessments`, { params: { requirement_id: requirementId } },
));
export const getQualityAnnotations = (sampleId, requirementId) => readData(apiClient.get(
  `${PREFIX}/samples/${encodeURIComponent(sampleId)}/quality-annotations`, { params: { requirement_id: requirementId } },
));
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

export const assessSample = (plantId, sampleId, requirementId, aiEnabled) => readData(apiClient.post(
  `${PREFIX}/assessments`,
  { plant_id: plantId, sample_id: sampleId, requirement_id: requirementId, ai_enabled: aiEnabled },
  { timeout: 120_000 },
));
