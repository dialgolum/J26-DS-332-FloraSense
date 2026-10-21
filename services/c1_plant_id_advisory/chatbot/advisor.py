import requests, chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"          # must match build_index.py
LLM = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM = (
    "You are the FloraSense plant advisor for a plant nursery. "
    "Answer ONLY from the CONTEXT below. If the context does not contain the answer, say you do not "
    "have that information and suggest asking a supervisor. "
    "Reply in the same language as the question (Sinhala or English). "
    "Keep plant and variety names in English. Be short and practical."
)

class Advisor:
    def __init__(self):
        self.emb = SentenceTransformer(EMBED_MODEL)
        self.col = chromadb.PersistentClient(path="chroma_db").get_collection("flora_docs")

    def retrieve(self, question, variety=None, k=4):
        query = f"{variety}: {question}" if variety else question
        r = self.col.query(
            query_embeddings=self.emb.encode([query], normalize_embeddings=True).tolist(), n_results=k)
        return r["documents"][0], r["metadatas"][0]

    def answer(self, question, variety=None):
        docs, metas = self.retrieve(question, variety)
        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
        plant = f"The plant in the photo is: {variety}.\n" if variety else ""
        r = requests.post(OLLAMA_URL, json={
            "model": LLM, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{plant}CONTEXT:\n{context}\n\nQUESTION: {question}"}],
            "options": {"temperature": 0.2, "num_ctx": 2048}}, timeout=300)
        data = r.json()
        if "message" not in data:
            raise RuntimeError(f"Ollama error: {data.get('error', data)}")
        sources = sorted({f"{m['source']} p{m['page']}" for m in metas})
        return data["message"]["content"], sources