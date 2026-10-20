import fitz, pandas as pd, chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCS = Path("data/raw/company_documents")
emb = SentenceTransformer("BAAI/bge-m3")
col = chromadb.PersistentClient(path="chroma_db").get_or_create_collection(
    "flora_docs", metadata={"hnsw:space": "cosine"})

def chunks(text, size=800, overlap=150):
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap

ids, texts, metas = [], [], []
for pdf in DOCS.glob("*.pdf"):
    for pno, page in enumerate(fitz.open(pdf), 1):
        for j, c in enumerate(chunks(page.get_text())):
            if c.strip():
                ids.append(f"{pdf.stem}-p{pno}-{j}"); texts.append(c)
                metas.append({"source": pdf.name, "page": pno})
for xl in DOCS.glob("*.xlsx"):
    for i, row in pd.read_excel(xl).fillna("").iterrows():
        ids.append(f"{xl.stem}-r{i}")
        texts.append("; ".join(f"{k}: {v}" for k, v in row.items() if v != ""))
        metas.append({"source": xl.name, "page": int(i)})

col.upsert(ids=ids, documents=texts, metadatas=metas,
           embeddings=emb.encode(texts, normalize_embeddings=True).tolist())
print(len(ids), "chunks indexed")

def ask(q, k=4):
    r = col.query(query_embeddings=emb.encode([q], normalize_embeddings=True).tolist(), n_results=k)
    for d, m in zip(r["documents"][0], r["metadatas"][0]):
        print(f"[{m['source']} p{m['page']}] {d[:150]}...\n")

ask("How should Aglaonema be packed for export?")