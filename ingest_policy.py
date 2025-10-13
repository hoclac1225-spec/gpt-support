# ingest_policy.py — enhanced: incremental update, VN-aware chunking, atomic writes
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

"""
Features vs your original:
- Incremental updates (avoid re-embedding unchanged chunks via text hash)
- Supports single file or a whole directory of *.md
- Vietnamese-aware chunking (., !, ?, …, ;, :), preserves code fences
- Safe atomic writes for both JSON and FAISS index
- CLI flags: --file/--dir, --rebuild, --maxlen, --model, --batch
- Stores richer meta: id, text, hash, ts, source
- Cosine similarity via L2-normalized vectors + IndexFlatIP (same as before)

Drop-in compatible with app.py that expects FAISS IndexFlatIP and meta json.
"""

import os, re, json, time, hashlib, argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===== ENV (kept compatible) =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATA_DIR       = os.getenv("DATA_DIR", "./data")
VECTOR_DIR     = os.getenv("VECTOR_DIR", "./vectors")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
POLICY_FILE    = os.getenv("POLICY_FILE", "policies.md")
POLICY_MAXLEN  = int(os.getenv("POLICY_MAXLEN", "900"))
EMBED_BATCH    = int(os.getenv("EMBED_BATCH", "128"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY chưa được cấu hình trong .env")

os.makedirs(VECTOR_DIR, exist_ok=True)
client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_PATH = str(Path(VECTOR_DIR) / "policies.index")
META_PATH  = str(Path(VECTOR_DIR) / "policies.meta.json")

# ===== Helpers =====
def _clean(txt: str) -> str:
    return re.sub(r"[ \t]+", " ", (txt or "")).strip()

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.M)
_HDR_SPLIT_RE  = re.compile(r"\n(?=#+\s)")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?…;:])\s+")  # VN-aware sentence-ish split


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _extract_blocks(md_text: str) -> List[str]:
    """Extract blocks while preserving fenced code blocks as atomic units."""
    if not md_text:
        return ["Default policy"]

    # Temporarily replace code fences with placeholders
    fences = []
    def _keep_fence(m):
        fences.append(m.group(0))
        return f"@@FENCE_{len(fences)-1}@@"

    tmp = _CODE_FENCE_RE.sub(_keep_fence, md_text)

    # Split by heading boundaries
    parts = _HDR_SPLIT_RE.split(tmp)
    blocks = []
    for p in parts:
        # Restore fences
        def _restore_fence(m):
            idx = int(m.group(1))
            return fences[idx]
        p = re.sub(r"@@FENCE_(\d+)@@", _restore_fence, p)
        p = _clean(p)
        if p:
            blocks.append(p)
    return blocks or ["Default policy"]


def smart_chunks(md_text: str, maxlen: int) -> List[str]:
    """
    Heading-first, then sentence-based, then hard cut. Preserves fenced code.
    """
    blocks = _extract_blocks(md_text)
    chunks: List[str] = []
    for b in blocks:
        if len(b) <= maxlen:
            chunks.append(b)
            continue
        # Sentence soft cut
        cur = ""
        for s in _SENT_SPLIT_RE.split(b):
            s = s.strip()
            if not s:
                continue
            if len(cur) + len(s) + 1 <= maxlen:
                cur += (" " if cur else "") + s
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = s
        if cur:
            chunks.append(cur.strip())

        # Final hard-safety if any remain too long
        final = []
        for c in chunks[-5:]:  # only check recent additions for oversize
            if len(c) > maxlen:
                for i in range(0, len(c), maxlen):
                    final.append(c[i:i+maxlen])
            else:
                final.append(c)
        chunks = chunks[:-5] + final

    return chunks or ["Default policy"]


def embed_batch(texts: List[str], batch_size: int) -> List[np.ndarray]:
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([np.asarray(e.embedding, dtype="float32") for e in resp.data])
    return vecs


def read_sources(file_path: Path = None, dir_path: Path = None) -> List[Tuple[str, str]]:
    """Return list of tuples (source, text)."""
    out: List[Tuple[str, str]] = []
    if dir_path and dir_path.exists():
        for p in sorted(dir_path.glob("*.md")):
            out.append((str(p), p.read_text(encoding="utf-8")))
    elif file_path and file_path.exists():
        out.append((str(file_path), file_path.read_text(encoding="utf-8")))
    else:
        # fallback to default
        p = Path(DATA_DIR) / POLICY_FILE
        if p.exists():
            out.append((str(p), p.read_text(encoding="utf-8")))
        else:
            print(f"⚠️  Không tìm thấy file policy. Dùng nội dung mặc định.")
            out.append(("default", "Default policy"))
    return out


def build_chunks_with_meta(sources: List[Tuple[str, str]], maxlen: int) -> List[Dict]:
    """Return list of chunk meta dicts with text & hash."""
    records: List[Dict] = []
    ts = int(time.time())
    for src, text in sources:
        for chunk in smart_chunks(text, maxlen=maxlen):
            chunk = _clean(chunk)
            if not chunk:
                continue
            records.append({
                "type": "policy",
                "text": chunk,
                "hash": _hash_text(chunk),
                "ts": ts,
                "source": src,
            })
    return records


def atomic_write_json(path: str, obj) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_index(path: str, index: faiss.Index) -> None:
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)


def load_existing() -> Tuple[faiss.Index, List[Dict]]:
    index = None
    meta: List[Dict] = []
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
        except Exception as e:
            print(f"⚠️  Không đọc được index cũ: {e}. Sẽ rebuild.")
            index = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"⚠️  Không đọc được meta cũ: {e}. Sẽ rebuild meta.")
            meta = []
    return index, meta


def ensure_index(d: int) -> faiss.Index:
    idx = faiss.IndexFlatIP(d)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Embed/update policy knowledge base")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn 1 file .md")
    parser.add_argument("--dir", type=str, default=None, help="Thư mục chứa *.md")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild toàn bộ thay vì incremental")
    parser.add_argument("--maxlen", type=int, default=POLICY_MAXLEN, help="Độ dài chunk tối đa")
    parser.add_argument("--model", type=str, default=EMBED_MODEL, help="Model embedding")
    parser.add_argument("--batch", type=int, default=EMBED_BATCH, help="Batch size embedding")
    args = parser.parse_args()

    global EMBED_MODEL
    EMBED_MODEL = args.model

    file_path = Path(args.file) if args.file else None
    dir_path  = Path(args.dir) if args.dir else None

    sources = read_sources(file_path=file_path, dir_path=dir_path)
    new_records = build_chunks_with_meta(sources, maxlen=args.maxlen)

    # Load existing index/meta for incremental
    index, meta = load_existing()

    # Decide what to embed
    if args.rebuild or index is None or not meta:
        print(f"🧩 Rebuild index từ {len(new_records)} chunks (max_len={args.maxlen})")
        texts = [r["text"] for r in new_records]
        vecs = embed_batch(texts, batch_size=args.batch)
        X = np.vstack(vecs).astype("float32")
        faiss.normalize_L2(X)
        idx = ensure_index(X.shape[1])
        idx.add(np.ascontiguousarray(X))

        # Renumber ids 0..n-1
        for i, r in enumerate(new_records):
            r["id"] = i
        atomic_write_index(INDEX_PATH, idx)
        atomic_write_json(META_PATH, new_records)
        print(f"✅ Rebuild xong. chunks={len(new_records)} -> {INDEX_PATH} / {META_PATH}")
        return

    # Incremental: add only chunks with new hash
    old_hashes = {r.get("hash") for r in meta}
    add_records = [r for r in new_records if r["hash"] not in old_hashes]

    if not add_records:
        print("ℹ️  Không có thay đổi chính sách nào (no new chunks).")
        return

    print(f"➕ Thêm mới {len(add_records)} chunks (trên tổng {len(new_records)})")
    texts = [r["text"] for r in add_records]
    vecs = embed_batch(texts, batch_size=args.batch)
    X = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(X)

    # If dims mismatch, force rebuild
    if index.d != X.shape[1]:
        print("⚠️  Dim không khớp với index cũ. Tiến hành rebuild.")
        all_texts = [r["text"] for r in new_records]
        vecs = embed_batch(all_texts, batch_size=args.batch)
        X = np.vstack(vecs).astype("float32")
        faiss.normalize_L2(X)
        index = ensure_index(X.shape[1])
        index.add(np.ascontiguousarray(X))
        for i, r in enumerate(new_records):
            r["id"] = i
        atomic_write_index(INDEX_PATH, index)
        atomic_write_json(META_PATH, new_records)
        print(f"✅ Rebuild xong do dim mismatch. chunks={len(new_records)}")
        return

    # Append to existing index
    start_id = len(meta)
    index.add(np.ascontiguousarray(X))
    for i, r in enumerate(add_records):
        r["id"] = start_id + i

    # Persist
    new_meta = meta + add_records
    atomic_write_index(INDEX_PATH, index)
    atomic_write_json(META_PATH, new_meta)

    print(f"✅ Đã thêm {len(add_records)} chunks. Tổng={len(new_meta)} -> {INDEX_PATH} / {META_PATH}")


if __name__ == "__main__":
    main()
