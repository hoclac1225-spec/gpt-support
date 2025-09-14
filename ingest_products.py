# ingest_products.py — bền vững + giàu metadata (SYNC với app.py)
# -*- coding: utf-8 -*-

import os, re, json, time, html, requests
import numpy as np, faiss
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse, parse_qs, quote

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
STORE            = os.getenv("SHOPIFY_STORE", "")              # ví dụ: xxx.myshopify.com
TOKEN            = os.getenv("SHOPIFY_ADMIN_API_TOKEN", "")
VECTOR_DIR       = os.getenv("VECTOR_DIR", "./vectors")
SHOP_FRONT_URL   = os.getenv("SHOP_FRONT_URL", "")             # ví dụ: https://shop.aloha.id.vn
FETCH_METAFIELDS = os.getenv("FETCH_METAFIELDS", "false").lower() == "true"
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # khớp app.py
EMBED_BATCH      = int(os.getenv("EMBED_BATCH", "128"))
API_VER          = os.getenv("SHOPIFY_API_VERSION", "2023-10")
# locales cần kéo bản dịch (ví dụ: zh, zh-TW, zh-CN, th, id…)
LOCALES          = [s.strip() for s in os.getenv("LOCALES", "zh,zh-TW,zh-CN").split(",") if s.strip()]

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

def fetch_product_translations(pid: int, locales):
    """
    Dùng Translations API để lấy title/body_html theo từng locale.
    Trả về: {locale: {"title": "...", "body_html": "..."}, ...}
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
# --- thêm gần phần fetch_* ---
TRAN_LOCALES = [s.strip() for s in os.getenv("TRAN_LOCALES", "zh-CN,zh-TW,vi,en,th,id").split(",")]

def fetch_translations(pid: int, locales=TRAN_LOCALES):
    """Lấy title/body/options/tags đã dịch từ Shopify Translations API."""
    outs = []
    for loc in locales:
        try:
            u = f"https://{STORE}/admin/api/{API_VER}/translations.json"
            params = {"locale": loc, "resource_type": "Product", "resource_id": pid}
            r = _get(u, params=params)
            arr = r.json().get("translations", []) or []
            # gom nội dung hữu ích
            for t in arr:
                k = (t.get("key") or "").lower()
                v = (t.get("value") or "").strip()
                if not v: 
                    continue
                if any(x in k for x in ["title","body_html","product_type","tags","option","name","value"]):
                    outs.append(v)
        except Exception as e:
            print("⚠️ translation fetch error pid=", pid, "loc=", loc, repr(e))
    # lọc rác + rút gọn
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
    # Triggers (có trong title/tags/body) -> các alias nên thêm vào tags
    "千層": ["千层", "mille crepe", "crepe", "可麗餅", "可丽饼", "เครป", "crepe cake"],
    "榴槤": ["榴莲", "durian", "ทุเรียน"],
    "可麗餅": ["可丽饼", "crepe", "mille crepe", "เครป"],
    "奶茶": ["milk tea", "bubble tea", "boba", "ชานม", "ชานมไข่มุก", "teh susu", "boba tea"],
    # bạn có thể mở rộng thêm…
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
    # khử trùng lặp, trả dạng ' | ' join được
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

        # options map: {"Color": ["Red","Blue"], ...}
        options_map = {(opt.get("name") or "").strip(): (opt.get("values") or [])
                       for opt in (p.get("options") or [])}

        # ảnh đại diện
        first_image = ""
        if p.get("images"):
            first_image = (p["images"][0].get("src") or "").strip()

        # metafields (nếu bật)
        specs_txt = fetch_product_metafields(p.get("id"))

        # ✅ tạo blob để dò trigger & bơm alias đa ngôn ngữ vào tags
        trigger_blob = " ".join([title, tags_base, body, specs_txt])
        alias_list = _expand_aliases(trigger_blob)
        alias_str  = " | ".join(alias_list) if alias_list else ""
        tags = " | ".join([t for t in [tags_base, alias_str] if t]).strip()

        # các biến dùng chung khi duyệt variants
        variants = p.get("variants") or [{}]
        first_price_fallback = None

        for v in variants:
            # ---- fields cấp variant ----
            sku   = (v.get("sku") or "").strip()
            price = (v.get("price") or "").strip()
            try:
                # nếu muốn giữ số để app dễ parse, có thể dùng float
                # nhưng ở đây ta giữ nguyên str và để app format lại
                pass
            except Exception:
                pass

            qty = v.get("inventory_quantity")
            try:
                qty = int(qty) if qty is not None else None
            except Exception:
                qty = None

            # available: ưu tiên flag của variant, sau đó dựa vào tồn kho/status
            available = bool(v.get("available", True))
            if qty is not None:
                available = available and (qty > 0)
            if status and status != "active":
                available = False

            # dùng price đầu tiên làm fallback cho các chunk/variant khác nếu trống
            if first_price_fallback is None and price:
                first_price_fallback = price

            # caption variant (ví dụ: "Color: Red | Size: M")
            variant_caption = (v.get("title") or "").strip()
            if not variant_caption or variant_caption.lower() == "default title":
                # ghép từ options để rõ ràng hơn
                pieces = []
                for k in options_map.keys():
                    # Shopify lưu value thật trong v.get('option1'|'option2'|'option3')
                    # map theo thứ tự Option1 -> option1, etc.
                    idx = len(pieces) + 1
                    val = (v.get(f"option{idx}") or "").strip()
                    if val:
                        pieces.append(f"{k}: {val}")
                variant_caption = " | ".join(pieces) if pieces else "Default"

            # ---- build base text cho RAG ----
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

            # ---- băm thành chunk & append meta ----
            for chunk in text_to_chunks(base, maxlen=800):
                docs.append({
                    "type": "product",
                    "id": p.get("id"),
                    "title": title or "",
                    "handle": handle or "",
                    "tags": tags or "",
                    "url": url or "",
                    "price": (price or first_price_fallback or ""),     # giữ dạng str
                    "product_type": ptype or "",
                    "vendor": vendor or "",
                    "status": status or "active",
                    "sku": sku or "",
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
