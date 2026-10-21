import { useEffect, useRef, useState } from "react";
import { MessageCircle, Send } from "lucide-react";
import Card from "../../components/ui/Card.jsx";
import Button from "../../components/ui/Button.jsx";
import { askAdvisor } from "../../api/chat.js";

const nice = (s) => s.replace(" | ", " ").replace(/_/g, " ");

export default function ChatPanel({ variety }) {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Ask me about plant care or export preparation. Identify a plant first for variety-specific answers." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef(null);

  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
    if (variety) setMessages((m) => [...m, { role: "bot", text: `I identified this plant as ${nice(variety)}. What would you like to know?` }]);
  }, [variety]);

  async function send(q) {
    const question = (q ?? input).trim();
    if (!question || busy) return;
    setInput(""); setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }, { role: "bot", text: "Thinking..." }]);
    let reply;
    try {
      const r = await askAdvisor(question, variety ? nice(variety) : null);
      reply = { role: "bot", text: r.answer, sources: r.sources };
    } catch {
      reply = { role: "bot", text: "Sorry, the advisor is not available. Check that Ollama and the backend are running." };
    }
    setMessages((m) => [...m.slice(0, -1), reply]);
    setBusy(false);
  }

  return (
    <Card className="p-5 sm:p-6">
      <div className="mb-4 flex items-center gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><MessageCircle size={19} /></span>
        <div>
          <p className="font-display font-extrabold text-forest">Plant advisor</p>
          <p className="text-xs text-muted">Answers come from the plant knowledge base</p>
        </div>
      </div>
      <div className="flex max-h-80 flex-col gap-3 overflow-y-auto rounded-xl bg-[#f7f9f6] p-4">
        {messages.map((m, i) => (
          <div key={i} className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${m.role === "user" ? "self-end bg-forest text-white" : "self-start border border-line bg-white text-ink"}`}>
            <span className="whitespace-pre-wrap">{m.text}</span>
            {m.sources?.length > 0 && <p className="mt-1 text-[11px] text-muted">Sources: {m.sources.join(", ")}</p>}
          </div>
        ))}
        <div ref={end} />
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="secondary" className="min-h-8 px-3 text-xs" disabled={!variety || busy}
          onClick={() => send("How do I care for this plant? Include light, watering and temperature.")}>Care tips</Button>
        <Button variant="secondary" className="min-h-8 px-3 text-xs" disabled={!variety || busy}
          onClick={() => send("How should this plant be prepared and graded for export?")}>Export checks</Button>
      </div>
      <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Type your question in English or Sinhala"
          className="min-h-11 flex-1 rounded-xl border border-line bg-white px-3 text-sm outline-none focus:border-leaf" />
        <Button type="submit" disabled={busy || !input.trim()}><Send size={16} />Send</Button>
      </form>
    </Card>
  );
}