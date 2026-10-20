export async function identifyPlant(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/identify", { method: "POST", body: fd });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}