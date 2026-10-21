import fitz, pandas as pd, chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCS = Path("data/raw/company_documents")
EMBED_MODEL = "BAAI/bge-m3"          # or "intfloat/multilingual-e5-small" for a lighter model
FOOTER = ("SAMPLE DEVELOPMENT DATA", "Page ")

emb = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("flora_docs")      # start clean so old chunks never linger
except Exception:
    pass
col = client.get_or_create_collection("flora_docs", metadata={"hnsw:space": "cosine"})


def pdf_chunks(pdf):
    """One chunk per paragraph block; short blocks (headings, table cells) join the next one."""
    for pno, page in enumerate(fitz.open(pdf), 1):
        buf = ""
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0:                        # skip image blocks
                continue
            t = " ".join(b[4].split())
            if not t or t.startswith(FOOTER):    # skip empty text and page footers
                continue
            if len(t) < 120:
                buf = (buf + " " + t).strip()
                if len(buf) > 700:
                    yield pno, buf
                    buf = ""
                continue
            yield pno, (buf + " " + t).strip()
            buf = ""
        if buf:
            yield pno, buf


ids, texts, metas = [], [], []

for pdf in DOCS.glob("*.pdf"):
    for j, (pno, c) in enumerate(pdf_chunks(pdf)):
        ids.append(f"{pdf.stem}-p{pno}-{j}")
        texts.append(c)
        metas.append({"source": pdf.name, "page": pno})

for xl in DOCS.glob("*.xlsx"):
    for i, row in pd.read_excel(xl).fillna("").iterrows():
        ids.append(f"{xl.stem}-r{i}")
        texts.append("; ".join(f"{k}: {v}" for k, v in row.items() if v != ""))
        metas.append({"source": xl.name, "page": int(i)})

if not ids:
    raise SystemExit(f"No PDF or Excel files found in {DOCS}")

col.upsert(ids=ids, documents=texts, metadatas=metas,
           embeddings=emb.encode(texts, normalize_embeddings=True, batch_size=16).tolist())
print(len(ids), "chunks indexed")


def ask(q, k=4, show=True):
    r = col.query(query_embeddings=emb.encode([q], normalize_embeddings=True).tolist(), n_results=k)
    if show:
        for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            print(f"[{m['source']} p{m['page']}] dist={dist:.3f}\n{d}\n")
    return r["documents"][0]


# ---- retrieval test: does the right fact appear in the top 4 chunks? ----
tests_en = [
    ("How much shade does Siam Aurora need?", "50-60% shade"),
    ("Why are spider plant leaf tips turning brown?", "fluoride"),
    ("What is the cold limit for croton?", "15 C"),
    ("Can Aglaonema Maria grow in low light?", "60-75% shade"),
    ("What makes Dracaena sanderiana Gold stems turn yellow?", "direct sun"),
    ("How should plants be prepared for packing?", "ventilated"),
]
tests_si = [
    ("සියම් ඕරෝරා ශාකයට කොපමණ සෙවනක් අවශ්‍යද?", "50-60% shade"),
    ("ස්පයිඩර් ප්ලාන්ට් එකේ කොළ අග දුඹුරු වෙන්නේ ඇයි?", "fluoride"),
    ("ක්‍රෝටන් ශාකයට දරාගත හැකි අවම උෂ්ණත්වය කීයද?", "15 C"),
    ("මරියා ඇග්ලෝනීමා අඩු ආලෝකයේ වැඩෙනවාද?", "60-75% shade"),
    ("ලක්කි බම්බු දඬු කහ පාට වෙන්නේ ඇයි?", "direct sun"),
    ("අපනයනය සඳහා ශාක ඇසිරීමට පෙර සූදානම් කරන්නේ කෙසේද?", "ventilated"),
]

def run_tests(tests, name):
    if not tests:
        return
    hits = 0
    for q, must in tests:
        docs = ask(q, show=False)
        ok = any(must.lower() in d.lower() for d in docs)
        hits += ok
        print("HIT " if ok else "MISS", "-", q)
    print(f"{name} hit@4: {hits}/{len(tests)}\n")

run_tests(tests_en, "English")
run_tests(tests_si, "Sinhala")