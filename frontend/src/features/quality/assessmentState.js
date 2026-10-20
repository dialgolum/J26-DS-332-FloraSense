export function measurementAvailability({ loading, error, sessions, heightAnnotations }) {
  if (loading) return "Loading measurement availability…";
  if (error) return "Measurement availability unavailable";
  if (sessions.length) return "Saved image-based measurements available";
  if (heightAnnotations.length) return "Manual Annotation";
  return "Measurement Required";
}

export function missingImageViews(images) {
  return ["Front", "Root", "Side", "Focus"].filter(
    (view) => !images.some((image) => image.View_Type === view),
  );
}
