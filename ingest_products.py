# ingest_products.py — bền vững + giàu metadata (SYNC với app.py)
# -*- coding: utf-8 -*-

import os, re, json, time, html, requests
import numpy as np, faiss
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse, parse_qs, quote
import hashlib

# --- Load .env TRƯỚC khi đọc os.getenv
load_dotenv()

# ==== Config ====
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
STORE            = os.getenv("SHOPIFY_STORE", "")              # ví dụ: xxx.myshopify.com
TOKEN            = os.getenv("SHOPIFY_ADMIN_API_TOKEN", "")
VECTOR_DIR       = os.getenv("VECTOR_DIR", "./vectors")
SHOP_FRONT_URL   = os.getenv("SHOP_FRONT_URL", "")             # ví dụ: https://shop.aloha.id.vn
FETCH_METAFIELDS = os.getenv("FETCH_METAFIELDS", "false").lower() == "true"
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # khớp app.py
EMBED_BATCH      = int(os.getenv("EMBED_BATCH", "128"))
API_VER          = os.getenv("SHOPIFY_API_VERSION", "2023-10")

# ✅ mở rộng đa ngôn ngữ mặc định
LOCALES = [s.strip() for s in os.getenv(
    "LOCALES",
    "zh,zh-CN,zh-TW,ja,ko,th,id,vi,en,fr,es,de,pt,ru"
).split(",") if s.strip()]
TRAN_LOCALES = [s.strip() for s in os.getenv(
    "TRAN_LOCALES",
    "zh,zh-CN,zh-TW,ja,ko,th,id,vi,en,fr,es,de,pt,ru"
).split(",")]

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))  # 900–1200 là hợp lý

# Chống rác/timeout khi nhúng
MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "60"))                # bỏ chunk quá ngắn
EMBED_RETRIES = int(os.getenv("EMBED_RETRIES", "3"))               # số lần retry batch
EMBED_RETRY_BASE_DELAY = float(os.getenv("EMBED_RETRY_BASE_DELAY", "1.2"))

os.makedirs(VECTOR_DIR, exist_ok=True)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY chưa được cấu hình trong .env")
if not STORE or not TOKEN:
    raise RuntimeError("Cần SHOPIFY_STORE và SHOPIFY_ADMIN_API_TOKEN trong .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- helpers ----------
def strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return html.unescape(s)

def text_to_chunks(txt, maxlen=CHUNK_MAX_CHARS, min_chars=MIN_CHARS):
    """Cắt đoạn + lọc tối thiểu ký tự để tránh chunk rỗng."""
    t = re.sub(r"\s+", " ", (txt or "")).strip()
    if len(t) < min_chars:
        return []
    return [t[i:i+maxlen] for i in range(0, len(t), maxlen)]

def _canon(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _fingerprint(s: str) -> str:
    return hashlib.sha1(_canon(s).encode("utf-8")).hexdigest()

def filter_docs(docs, min_chars=MIN_CHARS):
    """Lọc doc rỗng/quá ngắn & chuẩn hoá whitespace."""
    out = []
    for d in docs:
        t = _canon(d.get("text", ""))
        if len(t) < min_chars:
            continue
        dd = dict(d)
        dd["text"] = t
        out.append(dd)
    return out

def embed_batch(texts, batch_size=128, retries=EMBED_RETRIES):
    """
    Nhúng theo batch (retry khi lỗi) và degrade xuống từng item nếu batch fail.
    Trả về: (vectors, keep_mask) với keep_mask[i]=True nếu item i nhúng ok.
    """
    vectors, keep = [], [False] * len(texts)
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        chunk = texts[start:end]
        # thử batch trước
        done = False
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
                for j, e in enumerate(resp.data):
                    vectors.append(np.array(e.embedding, dtype="float32"))
                    keep[start + j] = True
                done = True
                break
            except Exception:
                if attempt + 1 < retries:
                    time.sleep(EMBED_RETRY_BASE_DELAY * (attempt + 1))
        if done:
            continue
        # degrade từng item
        for j, t in enumerate(chunk):
            try:
                r1 = client.embeddings.create(model=EMBED_MODEL, input=[t])
                vectors.append(np.array(r1.data[0].embedding, dtype="float32"))
                keep[start + j] = True
            except Exception:
                pass
    return vectors, keep

def shop_product_url(handle: str) -> str:
    handle = (handle or "").strip()
    if SHOP_FRONT_URL:
        base = SHOP_FRONT_URL.rstrip("/")
        return f"{base}/products/{quote(handle)}"
    base = STORE.strip().rstrip("/")
    if base.endswith(".myshopify.com"):
        shop = base.split(".myshopify.com")[0]
        return f"https://{shop}.myshopify.com/products/{quote(handle)}"
    return f"https://{base}/products/{quote(handle)}"

def atomic_write_json(path: str, obj):
    """Ghi file JSON an toàn tránh hỏng file giữa chừng."""
    data = json.dumps(obj, ensure_ascii=False)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)

# ---------- Shopify session + retry 429 ----------
S = requests.Session()
S.headers.update({
    "X-Shopify-Access-Token": TOKEN,
    "Accept": "application/json",
    "Content-Type": "application/json",
})

def _get(url, params=None):
    while True:
        r = S.get(url, params=params, timeout=40)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = int(r.headers.get("Retry-After", "2"))
        print(f"⚠️ 429 rate-limited. Sleep {wait}s…")
        time.sleep(wait)

# ---------- fetch ----------
def fetch_all_products():
    """Lấy tất cả sản phẩm với phân trang page_info (Shopify REST)."""
    url = f"https://{STORE}/admin/api/{API_VER}/products.json"
    params = {"limit": 250}
    products = []

    while True:
        r = _get(url, params=params)
        data = r.json().get("products", [])
        products.extend(data)

        link = r.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if not m:
            break
        next_url = m.group(1)

        u = urlparse(next_url)
        url = f"{u.scheme}://{u.netloc}{u.path}"
        q = parse_qs(u.query)
        params = {k: (v[0] if isinstance(v, list) and v else "") for k, v in q.items()}

    print(f"✅ fetched {len(products)} products")
    return products

def fetch_product_metafields(pid: int):
    """Tuỳ chọn: kéo metafields mô tả specs."""
    if not FETCH_METAFIELDS:
        return ""
    url = f"https://{STORE}/admin/api/{API_VER}/products/{pid}/metafields.json"
    r = _get(url)
    mfs = r.json().get("metafields", [])
    parts = []
    for m in mfs:
        key = m.get("key")
        val = m.get("value")
        ns  = m.get("namespace")
        if key and val:
            parts.append(f"{ns}.{key}: {val}")
    return " | ".join(parts)

def fetch_product_translations(pid: int, locales):
    """
    Dùng Translations API để lấy title/body_html theo từng locale.
    Trả về: {locale: {'title': '...', 'body_html': '...'}, ...}
    """
    out = {}
    if not locales:
        return out
    for lc in locales:
        try:
            url = f"https://{STORE}/admin/api/{API_VER}/translations.json"
            params = {"locale": lc, "resource_type": "Product", "resource_id": pid}
            r = _get(url, params=params)
            arr = r.json().get("translations", []) or []
            rec = {}
            for tr in arr:
                key = tr.get("key")
                val = tr.get("value")
                if key in ("title", "body_html") and val:
                    rec[key] = val
            if rec:
                out[lc] = rec
        except Exception as e:
            print(f"⚠️ translation fetch error pid={pid} lc={lc}: {repr(e)}")
    return out

def fetch_translations(pid: int, locales=TRAN_LOCALES):
    """Lấy title/body/options/tags đã dịch từ Shopify Translations API."""
    outs = []
    for loc in locales:
        try:
            u = f"https://{STORE}/admin/api/{API_VER}/translations.json"
            params = {"locale": loc, "resource_type": "Product", "resource_id": pid}
            r = _get(u, params=params)
            arr = r.json().get("translations", []) or []
            for t in arr:
                k = (t.get("key") or "").lower()
                v = (t.get("value") or "").strip()
                if not v:
                    continue
                if any(x in k for x in ["title","body_html","product_type","tags","option","name","value"]):
                    outs.append(v)
        except Exception as e:
            print("⚠️ translation fetch error pid=", pid, "loc=", loc, repr(e))
    text = " | ".join(re.sub(r"<[^>]+>"," ", s) for s in outs if s)
    return re.sub(r"\s+", " ", text).strip()

# ---------- build docs ----------
# --- Alias mở rộng: ZH (Phồn -> Giản) + TH + ID + EN ---
_ZH_T2S = {
    "千層": "千层",
    "榴槤": "榴莲",
    "可麗餅": "可丽饼",
}
KW_EXPANSIONS = {
    "千層": ["千层", "mille crepe", "crepe", "可麗餅", "可丽饼", "เครป", "crepe cake"],
    "榴槤": ["榴莲", "durian", "ทุเรียน"],
    "可麗餅": ["可丽饼", "crepe", "mille crepe", "เครป"],
    "奶茶": ["milk tea", "bubble tea", "boba", "ชานม", "ชานมไข่มุก", "teh susu", "boba tea"],
}

def _expand_aliases(text_blob: str) -> list[str]:
    """Nhìn cụm xuất hiện trong text -> trả về list alias để nhét thêm vào tags."""
    blob = (text_blob or "").lower()
    adds = []
    # map T->S đơn giản
    for t, s in _ZH_T2S.items():
        if t in text_blob:
            adds.append(s)
    # alias theo trigger
    for trig, alist in KW_EXPANSIONS.items():
        if trig in text_blob:
            adds.extend(alist)
    out, seen = [], set()
    for x in adds:
        xx = x.strip()
        if xx and xx not in seen:
            seen.add(xx); out.append(xx)
    return out

def build_docs(products):
    docs = []
    for p in products:
        # ---- fields cấp product ----
        title   = (p.get("title") or "").strip()
        handle  = (p.get("handle") or "").strip()
        url     = shop_product_url(handle)
        body    = strip_html(p.get("body_html", ""))
        tags_base = p.get("tags", "") or ""
        ptype   = p.get("product_type", "") or ""
        vendor  = p.get("vendor", "") or ""
        status  = (p.get("status") or "active").lower()

        created_at   = p.get("created_at")   or ""
        published_at = p.get("published_at") or ""
        updated_at   = p.get("updated_at")   or ""

        options_map = {(opt.get("name") or "").strip(): (opt.get("values") or [])
                       for opt in (p.get("options") or [])}

        first_image = ""
        if p.get("images"):
            first_image = (p["images"][0].get("src") or "").strip()

        specs_txt = fetch_product_metafields(p.get("id"))

        # 🔎 Lấy dịch theo mọi locale (title/body)
        trans_map = fetch_product_translations(p.get("id"), LOCALES)  # {locale: {'title','body_html'}}
        # trích riêng CJK title cho meta
        title_zh = (
            (trans_map.get("zh") or {}).get("title") or
            (trans_map.get("zh-CN") or {}).get("title") or
            (trans_map.get("zh-TW") or {}).get("title") or
            ""
        )
        title_ja = (trans_map.get("ja") or {}).get("title") or ""
        title_ko = (trans_map.get("ko") or {}).get("title") or ""

        # Gom khối dịch đa ngữ để embed
        ml_bits = []
        for loc, rec in (trans_map or {}).items():
            tloc = (rec.get("title") or "").strip()
            bloc = strip_html(rec.get("body_html") or "")
            if tloc or bloc:
                ml_bits.append(f"[{loc}] {tloc} | {bloc}")
        extra_trans = " | ".join(x for x in ml_bits if x).strip()

        # (tuỳ chọn) Gom thêm các bản dịch rải rác khác (tags/options/values…)
        trans_blob = fetch_translations(p.get("id"), TRAN_LOCALES)

        # ✅ alias đa ngôn ngữ vào tags (tính cả phần dịch)
        trigger_blob = " ".join([title, tags_base, body, specs_txt, extra_trans, trans_blob])
        alias_list = _expand_aliases(trigger_blob)
        alias_str  = " | ".join(alias_list) if alias_list else ""
        tags = " | ".join([t for t in [tags_base, alias_str] if t]).strip()

        variants = p.get("variants") or [{}]
        first_price_fallback = None

        for v in variants:
            sku   = (v.get("sku") or "").strip()
            price = (v.get("price") or "").strip()
            variant_id = v.get("id")

            qty = v.get("inventory_quantity")
            try:
                qty = int(qty) if qty is not None else None
            except Exception:
                qty = None

            available = bool(v.get("available", True))
            if qty is not None:
                available = available and (qty > 0)
            if status and status != "active":
                available = False

            if first_price_fallback is None and price:
                first_price_fallback = price

            variant_caption = (v.get("title") or "").strip()
            if not variant_caption or variant_caption.lower() == "default title":
                pieces = []
                for k in options_map.keys():
                    idx = len(pieces) + 1
                    val = (v.get(f"option{idx}") or "").strip()
                    if val:
                        pieces.append(f"{k}: {val}")
                variant_caption = " | ".join(pieces) if pieces else "Default"

            base = (
                f"Product: {title}\n"
                f"Type: {ptype}\nVendor: {vendor}\nTags: {tags}\n"
                f"Options: {json.dumps(options_map, ensure_ascii=False)}\n"
                f"Variant: {variant_caption}\n"
                f"SKU: {sku or '-'}\n"
                f"Price: {price or first_price_fallback or '-'}\n"
                f"Inventory: {qty if qty is not None else '-'}\n"
                f"Available: {available}\n"
                f"Status: {status}\n"
                f"{('Specs: ' + specs_txt) if specs_txt else ''}\n"
                f"Body: {body}\n"
                f"{('Translations_ML: ' + ' | '.join([extra_trans, trans_blob]).strip(' |')) if (extra_trans or trans_blob) else ''}\n"
                f"URL: {url}"
            )

            for chunk in text_to_chunks(base, maxlen=CHUNK_MAX_CHARS):
                docs.append({
                    "type": "product",
                    "id": p.get("id"),
                    "title": title or "",
                    "title_zh": title_zh or "",
                    "title_ja": title_ja or "",
                    "title_ko": title_ko or "",
                    "handle": handle or "",
                    "tags": tags or "",
                    "url": url or "",
                    "price": (price or first_price_fallback or ""),
                    "product_type": ptype or "",
                    "vendor": vendor or "",
                    "status": status or "active",
                    "sku": sku or "",
                    "variant_id": str(variant_id or ""),
                    "variant": variant_caption or "Default",
                    "image": first_image or "",
                    "inventory_quantity": qty if (qty is None or isinstance(qty, int)) else None,
                    "available": bool(available),
                    "created_at": created_at,
                    "published_at": published_at,
                    "updated_at": updated_at,
                    "text": chunk or ""
                })

    return docs

def dedup_docs(docs):
    """
    Dedup an toàn: giữ riêng từng variant.
    Key gồm: (product_id, variant_id/sku/variant_caption, hash(text)).
    """
    seen = set()
    uniq = []
    for d in docs:
        text_fp = _fingerprint(d.get("text", ""))
        key = (
            str(d.get("id") or ""),
            str(d.get("variant_id") or d.get("sku") or d.get("variant") or ""),
            text_fp,
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq

# ---------- save ----------
def save_faiss(docs, index_path, meta_path):
    if not docs:
        raise RuntimeError("No docs to index.")

    # Lọc & chuẩn hoá để tránh rác
    docs2 = filter_docs(docs, min_chars=MIN_CHARS)
    if not docs2:
        raise RuntimeError("All docs too short or empty after filtering.")

    print(f"🧱 building embeddings for {len(docs2)} candidate chunks…")
    vecs, keep = embed_batch([d["text"] for d in docs2], batch_size=EMBED_BATCH)

    good_meta = [d for d, k in zip(docs2, keep) if k]
    if not vecs or not good_meta:
        raise RuntimeError("Embedding failed for all inputs.")

    X = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X))

    faiss.write_index(index, index_path)
    atomic_write_json(meta_path, good_meta)

    skipped = len(docs2) - len(good_meta)
    print(f"💾 Saved {len(good_meta)} / {len(docs2)} chunks → {index_path}")
    print(f"📝 Meta: {meta_path} | skipped: {skipped}")
    return len(good_meta), skipped

# ---------- main ----------
if __name__ == "__main__":
    t0 = time.time()
    products = fetch_all_products()
    docs = build_docs(products)
    docs = dedup_docs(docs)
    saved, skipped = save_faiss(docs, f"{VECTOR_DIR}/products.index", f"{VECTOR_DIR}/products.meta.json")
    print(f"✅ Done in {time.time() - t0:.1f}s | saved={saved} skipped={skipped}")
