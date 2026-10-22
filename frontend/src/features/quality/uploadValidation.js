export function validateFrontImage(file) {
  if (!file) return "Choose a P01 front-view image.";
  if (!["image/jpeg", "image/png"].includes(file.type)) return "Choose a JPEG or PNG image.";
  if (file.size > 25 * 1024 * 1024) return "Image exceeds the 25 MiB limit.";
  if (file.size === 0) return "The image is empty.";
  return "";
}
