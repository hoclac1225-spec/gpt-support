# ingest_products.py — bền vững + giàu metadata (SYNC với app.py)
# -*- coding: utf-8 -*-

import os, re, json, time, html, requests
import numpy as np, faiss
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse, parse_qs, quote

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
STORE            = os.getenv("SHOPIFY_STORE", "")              # ví dụ: 9mn9fa-6p.myshopify.com
TOKEN            = os.getenv("SHOPIFY_ADMIN_API_TOKEN", "")
VECTOR_DIR       = os.getenv("VECTOR_DIR", "./vectors")
SHOP_FRONT_URL   = os.getenv("SHOP_FRONT_URL", "")             # ví dụ: https://shop.aloha.id.vn (tùy chọn)
FETCH_METAFIELDS = os.getenv("FETCH_METAFIELDS", "false").lower() == "true"
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # khớp app.py
EMBED_BATCH      = int(os.getenv("EMBED_BATCH", "128"))
API_VER          = os.getenv("SHOPIFY_API_VERSION", "2023-10")

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

def text_to_chunks(txt, maxlen=800):
    txt = re.sub(r"\s+", " ", (txt or "")).strip()
    return [txt[i:i+maxlen] for i in range(0, len(txt), maxlen)] or [""]

def embed_batch(texts, batch_size=128):
    """Nhúng theo batch để tránh payload lớn/timeout."""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    return vecs

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

# ---------- build docs ----------
def build_docs(products):
    docs = []
    for p in products:
        title   = (p.get("title") or "").strip()
        handle  = (p.get("handle") or "").strip()
        url     = shop_product_url(handle)
        body    = strip_html(p.get("body_html", ""))
        tags    = p.get("tags", "") or ""
        ptype   = p.get("product_type", "") or ""
        vendor  = p.get("vendor", "") or ""
        status  = (p.get("status") or "active").lower()  # 'active'|'draft'|'archived'

        options_map = {(opt.get("name") or "").strip(): opt.get("values", [])
                       for opt in (p.get("options") or [])}

        first_image = ""
        if p.get("images"):
            first_image = p["images"][0].get("src") or ""

        specs_txt = fetch_product_metafields(p.get("id"))

        variants = p.get("variants") or [{}]
        first_price_fallback = None

        for v in variants:
            sku   = (v.get("sku") or "").strip()
            price = v.get("price")
            if price is not None:
                price = str(price).strip()
            if first_price_fallback is None and price:
                first_price_fallback = price

            # inventory_quantity -> int | None
            try:
                qty_raw = v.get("inventory_quantity")
                qty = int(qty_raw) if qty_raw is not None else None
            except Exception:
                qty = None

            # available: suy diễn
            if qty is not None:
                available = (qty > 0) and (status == "active")
            else:
                available = (status == "active")

            # caption variant
            optvals = [v.get("option1"), v.get("option2"), v.get("option3")]
            optvals = [o for o in optvals if o]
            variant_caption = " / ".join(optvals) if optvals else (v.get("title") or "").strip() or "Default"

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
                f"URL: {url}"
            )

            for chunk in text_to_chunks(base, maxlen=800):
                docs.append({
                    "type": "product",
                    "id": p.get("id"),
                    "title": title or "",
                    "handle": handle or "",
                    "tags": tags or "",
                    "url": url or "",
                    "price": (price or first_price_fallback or ""),    # app.py hiển thị chuỗi + “ đ”
                    "product_type": ptype or "",
                    "vendor": vendor or "",
                    "status": status or "active",                      # 'active'|'draft'|'archived'
                    "sku": sku or "",
                    "variant": variant_caption or "Default",
                    "image": first_image or "",
                    "inventory_quantity": qty if (qty is None or isinstance(qty, int)) else None,
                    "available": bool(available),
                    "text": chunk or ""
                })
    return docs

def dedup_docs(docs):
    """Loại trùng lặp nội dung để giảm nhiễu index."""
    seen = set()
    uniq = []
    for d in docs:
        h = hash(d["text"])
        if h in seen:
            continue
        seen.add(h)
        uniq.append(d)
    return uniq

# ---------- save ----------
def save_faiss(docs, index_path, meta_path):
    if not docs:
        raise RuntimeError("No docs to index.")
    print(f"🧱 building embeddings for {len(docs)} chunks…")
    X = np.vstack(embed_batch([d["text"] for d in docs], batch_size=EMBED_BATCH)).astype("float32")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X))
    faiss.write_index(index, index_path)
    atomic_write_json(meta_path, docs)
    print(f"💾 Saved {len(docs)} chunks → {index_path}")
    print(f"📝 Meta: {meta_path}")

# ---------- main ----------
if __name__ == "__main__":
    t0 = time.time()
    products = fetch_all_products()
    docs = build_docs(products)
    docs = dedup_docs(docs)
    save_faiss(docs, f"{VECTOR_DIR}/products.index", f"{VECTOR_DIR}/products.meta.json")
    print(f"✅ Done in {time.time() - t0:.1f}s")
