import sys, requests, chromadb
from sentence_transformers import SentenceTransformer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma3:4b"
EMBED_MODEL = "BAAI/bge-m3"                       # must match build_index.py
emb = SentenceTransformer(EMBED_MODEL)
col = chromadb.PersistentClient(path="chroma_db").get_collection("flora_docs")

SYSTEM = (
    "You are the FloraSense plant advisor for a plant nursery. "
    "Answer ONLY from the CONTEXT below. If the context does not contain the answer, say you do not "
    "have that information and suggest asking a supervisor. "
    "Reply in the same language as the question (Sinhala or English). "
    "Keep plant and variety names in English. Be short and practical."
)

def retrieve(q, variety=None, k=4):
    query = f"{variety}: {q}" if variety else q   # the identified variety sharpens retrieval later
    r = col.query(query_embeddings=emb.encode([query], normalize_embeddings=True).tolist(), n_results=k)
    return r["documents"][0], r["metadatas"][0]

def answer(q, variety=None):
    docs, metas = retrieve(q, variety)
    context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {q}"}],
        "options": {"temperature": 0.2}}, timeout=300)
    data = r.json()
    if "message" not in data:
        raise RuntimeError(f"Ollama error: {data.get('error', data)}")
    return data["message"]["content"], metas

if __name__ == "__main__":
    print(f"Model: {MODEL}. Type a question (Ctrl+C to stop).")
    while True:
        q = input("\nYou: ")
        text, metas = answer(q)
        print("\nBot:", text)
        print("Sources:", ", ".join(sorted({f"{m['source']} p{m['page']}" for m in metas})))
        

