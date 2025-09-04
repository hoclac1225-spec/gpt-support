# ingest_policy.py — SYNC với app.py (FAISS IP + L2, EMBED_MODEL, batch, atomic)
# -*- coding: utf-8 -*-

import os, re, json, numpy as np, faiss
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===== ENV =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATA_DIR       = os.getenv("DATA_DIR", "./data")
VECTOR_DIR     = os.getenv("VECTOR_DIR", "./vectors")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # khớp app.py
POLICY_FILE    = os.getenv("POLICY_FILE", "policies.md")
POLICY_MAXLEN  = int(os.getenv("POLICY_MAXLEN", "900"))
EMBED_BATCH    = int(os.getenv("EMBED_BATCH", "128"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY chưa được cấu hình trong .env")

os.makedirs(VECTOR_DIR, exist_ok=True)
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== helpers =====
def _clean(txt: str) -> str:
    return re.sub(r"[ \t]+", " ", (txt or "")).strip()

def _smart_chunks(md_text: str, maxlen: int) -> list[str]:
    """
    Chia đoạn ưu tiên theo heading (#, ## ...), nếu dài thì cắt theo câu, rồi mới cắt cứng.
    """
    if not md_text:
        return ["Default policy"]

    blocks = re.split(r"\n(?=#+\s)", md_text, flags=re.M)
    chunks = []
    for b in blocks or [md_text]:
        b = _clean(b)
        if not b:
            continue
        if len(b) <= maxlen:
            chunks.append(b)
        else:
            # cắt mềm theo câu
            sentences = re.split(r"(?<=[\.!?])\s+", b)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) + 1 <= maxlen:
                    cur += (" " if cur else "") + s
                else:
                    if cur:
                        chunks.append(cur.strip())
                    cur = s
            if cur:
                chunks.append(cur.strip())

    return chunks or ["Default policy"]

def embed_batch(texts: list[str], batch_size: int = 128) -> list[np.ndarray]:
    vecs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch_size])
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    return vecs

def atomic_write_json(path: str, obj):
    data = json.dumps(obj, ensure_ascii=False)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)

# ===== main =====
if __name__ == "__main__":
    src_path = Path(DATA_DIR) / POLICY_FILE
    if not src_path.exists():
        print(f"⚠️  {src_path} không tồn tại. Dùng nội dung mặc định.")
        text = "Default policy"
    else:
        text = src_path.read_text(encoding="utf-8") or "Default policy"

    chunks = _smart_chunks(text, maxlen=POLICY_MAXLEN)
    print(f"🧩 policy chunks: {len(chunks)} (max_len={POLICY_MAXLEN})")

    X = np.vstack(embed_batch(chunks, batch_size=EMBED_BATCH)).astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X))

    faiss.write_index(index, f"{VECTOR_DIR}/policies.index")
    meta = [{"type": "policy", "id": i, "text": c} for i, c in enumerate(chunks)]
    atomic_write_json(f"{VECTOR_DIR}/policies.meta.json", meta)

    print(f"✅ Policies embedded. chunks={len(chunks)} -> {VECTOR_DIR}/policies.index / policies.meta.json")
