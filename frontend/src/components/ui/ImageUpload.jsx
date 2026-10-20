import { useEffect, useState } from "react";
import { ImagePlus, RotateCcw, Upload } from "lucide-react";
import Button from "./Button.jsx";

export default function ImageUpload({ label = "Plant image", onFileChange, accept = "image/*" }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");

  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(nextFile) {
    const imageFile = nextFile?.type?.startsWith("image/") ? nextFile : null;
    setFile(imageFile);
    onFileChange?.(imageFile);
  }

  return <div>
    <div className="mb-2 flex items-center justify-between"><label className="text-sm font-bold text-ink">{label}</label>{file && <Button variant="quiet" className="min-h-8 px-2 text-xs" onClick={() => selectFile(null)}><RotateCcw size={14} />Remove</Button>}</div>
    {!file ? <label className="group flex min-h-[260px] cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-[#cbd8ce] bg-[#fbfcfa] px-5 text-center transition hover:border-leaf/60 hover:bg-mint/25 sm:min-h-[310px]">
      <span className="mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-mint text-leaf transition group-hover:scale-105"><ImagePlus size={25} /></span><span className="font-bold text-forest">Choose an image to preview</span><span className="mt-1 text-xs text-muted">Select a plant image from this device</span><span className="mt-5 inline-flex items-center gap-2 rounded-xl border border-line bg-white px-3 py-2 text-xs font-bold text-forest"><Upload size={14} />Browse images</span>
      <input className="sr-only" type="file" accept={accept} onChange={(event) => selectFile(event.target.files?.[0])} />
    </label> : <div className="overflow-hidden rounded-2xl border border-line bg-[#f7f9f6]"><img src={preview} alt={`Selected ${label.toLowerCase()}`} className="max-h-[410px] min-h-[220px] w-full object-contain" /><div className="flex items-center justify-between gap-3 border-t border-line bg-white px-4 py-3"><span className="truncate text-xs font-semibold text-ink">{file.name}</span><span className="shrink-0 text-[11px] text-muted">{(file.size / 1024 / 1024).toFixed(2)} MB</span></div></div>}
  </div>;
}
