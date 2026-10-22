import { ArrowUp, Bot, MessageCircle, RotateCcw } from "lucide-react";
import { useState } from "react";
import Card from "../../components/ui/Card.jsx";
import PageIntro from "../../components/ui/PageIntro.jsx";
import Button from "../../components/ui/Button.jsx";
import Field, { Select } from "../../components/ui/Field.jsx";
import IntegrationNotice from "../../components/ui/IntegrationNotice.jsx";

export default function ChatbotPage() {
  const [language, setLanguage] = useState("English");
  const [draft, setDraft] = useState("");
  const [notice, setNotice] = useState(false);
  function submit(event) {
    event.preventDefault();
    if (draft.trim()) setNotice(true);
  }
  return <>
    <PageIntro eyebrow="C1 · Plant assistant" title="Plant care, in conversation" description="A workspace for plant-related questions. The chatbot, retrieval system, and conversation memory are not connected yet." />
    <Card className="mx-auto max-w-4xl overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-line px-5 py-4 sm:px-6"><div className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><Bot size={20} /></span><div><p className="font-bold text-forest">FloraSense assistant</p><p className="flex items-center gap-1.5 text-xs text-muted"><span className="h-1.5 w-1.5 rounded-full bg-amber-400" />Not connected</p></div></div><div className="flex items-center gap-2"><Field label="Language"><Select aria-label="Conversation language" value={language} onChange={(event) => setLanguage(event.target.value)}><option>English</option><option>Sinhala</option></Select></Field><Button variant="quiet" className="mt-6 min-h-9 px-2.5" onClick={() => { setDraft(""); setNotice(false); }} aria-label="Clear draft"><RotateCcw size={15} /></Button></div></div>
      <div className="flex min-h-[390px] flex-col items-center justify-center px-5 py-10 text-center sm:min-h-[440px]"><span className="grid h-14 w-14 place-items-center rounded-2xl bg-[#f1f6f0] text-leaf"><MessageCircle size={24} /></span><h3 className="mt-4 font-display text-lg font-extrabold text-forest">Your conversation starts here</h3><p className="mt-2 max-w-sm text-sm leading-6 text-muted">Ask about ornamental plants, growing conditions, or care. The assistant API will be connected when C1 conversation handling is ready.</p>{notice && <div className="mt-5 max-w-lg"><IntegrationNotice>Your message has not been sent. No assistant reply was generated.</IntegrationNotice></div>}</div>
      <form onSubmit={submit} className="border-t border-line bg-[#fbfcfa] p-4 sm:p-5"><label className="sr-only" htmlFor="chat-message">Message</label><div className="flex items-end gap-2 rounded-2xl border border-line bg-white p-2 shadow-sm focus-within:border-leaf/40 focus-within:ring-2 focus-within:ring-leaf/10"><textarea id="chat-message" rows="2" value={draft} onChange={(event) => { setDraft(event.target.value); setNotice(false); }} placeholder={`Write a plant question in ${language}…`} className="max-h-36 min-h-11 flex-1 resize-y border-0 bg-transparent px-2 py-2 text-sm outline-none placeholder:text-slate-400" /><Button type="submit" className="min-h-10 w-10 shrink-0 px-0" aria-label="Prepare message"><ArrowUp size={18} /></Button></div><p className="mt-2 px-1 text-[11px] text-muted">Messages remain local to this page; submission does not contact a model.</p></form>
    </Card>
  </>;
}
