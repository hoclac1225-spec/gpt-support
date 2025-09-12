# -*- coding: utf-8 -*-
import unicodedata
import os, json, time, re, requests, numpy as np, faiss, threading, random
from collections import deque
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import hmac, hashlib, base64

# --- text normalize helpers (có & không dấu)
def _strip_accents(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _normalize_text(s: str) -> str:
    # giữ lại chữ, số, khoảng trắng, các bảng chữ cái mở rộng
    return re.sub(r"[^0-9a-z\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff\u0E00-\u0E7F ]", " ", (s or "").lower()).strip()

def _norm_both(s: str):
    """Trả về tuple (có_dấu, không_dấu) đã normalize & lower."""
    n1 = _normalize_text(s)
    n2 = _normalize_text(_strip_accents(s))
    return n1, n2
# số từ tối thiểu phải trùng trong title (có thể cho vào ENV nếu muốn)
TITLE_MIN_WORDS = int(os.getenv("TITLE_MIN_WORDS", "2"))

def _has_title_overlap(q, hits, min_words: int = TITLE_MIN_WORDS, min_cover: float = 0.6):
    """
    Trả True nếu:
    - Có ít nhất 'min_words' từ trong câu hỏi xuất hiện trong title (đã normalize, có/không dấu), HOẶC
    - Tỷ lệ phủ từ (matched/len(tokens)) >= min_cover  (fallback cho câu rất ngắn / ngôn ngữ không có khoảng trắng)
    """
    qn1, qn2 = _norm_both(q)
    # tokens theo khoảng trắng, bỏ từ 1 ký tự
    qtok = [w for w in qn1.split() if len(w) > 1]
    if not qtok:                    # ví dụ tiếng Trung → không tách được từ
        qtok = [qn1]                # fallback: dùng cả chuỗi đã normalize

    for d in hits[:5]:
        t1, t2 = _norm_both(d.get("title", ""))
        matched = sum(1 for w in qtok if (w in t1) or (w in t2))

        # Điều kiện “ít nhất N từ trùng”
        cond_min_words = (len(qtok) >= min_words and matched >= min_words)
        # Fallback coverage (giữ logic cũ): hữu ích khi câu hỏi quá ngắn
        cond_cover = (matched / max(1, len(qtok))) >= min_cover

        if cond_min_words or cond_cover:
            return True
    return False




# ========= BOOTSTRAP =========
load_dotenv()
app = Flask(__name__)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
origins = [o.strip() for o in allowed_origins.split(",")] if allowed_origins else ["*"]
CORS(app, origins=origins)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN", "aloha_verify_123")
# ---- Multi-page map ----
# ---- Multi-id -> token (FB Page & IG) ----
TOKEN_MAP = {}

def _add_map(key, token):
    if key and token:
        TOKEN_MAP[str(key)] = token

for i in range(1, 11):
    pid = os.getenv(f"FB_PAGE_ID_{i}")
    ptk = os.getenv(f"FB_PAGE_TOKEN_{i}")
    _add_map(pid, ptk)  # map Page ID -> Page token

    igid = os.getenv(f"IG_ACCOUNT_ID_{i}")
    _add_map(igid, ptk) # map IG Account ID -> dùng CHUNG Page token đã liên kết IG
print("TOKEN_MAP size:", len(TOKEN_MAP))


VECTOR_DIR       = os.getenv("VECTOR_DIR", "./vectors")
# Shopify
SHOPIFY_SHOP = os.getenv("SHOPIFY_STORE", "")  # domain *.myshopify.com (tham chiếu)
# Link shop mặc định (fallback)
SHOP_URL         = os.getenv("SHOP_URL", "https://shop.aloha.id.vn/zh")
# Đa ngôn ngữ
SUPPORTED_LANGS  = [s.strip() for s in os.getenv("SUPPORTED_LANGS", "vi,en,zh,th,id").split(",")]
DEFAULT_LANG     = os.getenv("DEFAULT_LANG", "vi")
SHOP_URL_MAP = {
    "vi": os.getenv("SHOP_URL_VI", SHOP_URL),
    "en": os.getenv("SHOP_URL_EN", SHOP_URL),
    "zh": os.getenv("SHOP_URL_ZH", SHOP_URL),
    "th": os.getenv("SHOP_URL_TH", SHOP_URL),
    "id": os.getenv("SHOP_URL_ID", SHOP_URL),
}

REPHRASE_ENABLED = os.getenv("REPHRASE_ENABLED", "true").lower() == "true"
EMOJI_MODE       = os.getenv("EMOJI_MODE", "cute")  # "cute" | "none"

# Lọc & ngưỡng điểm
SCORE_MIN = float(os.getenv("PRODUCT_SCORE_MIN", "0.28"))
STRICT_MATCH = os.getenv("STRICT_MATCH", "true").lower() == "true"

print("=== BOOT ===")
print("VECTOR_DIR:", os.path.abspath(VECTOR_DIR))
print("TOKEN_MAP size:", len(TOKEN_MAP))
print("OPENAI key set:", bool(OPENAI_API_KEY))




# OpenAI
OPENAI_URL   = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-3-small")

oai_client   = OpenAI(api_key=OPENAI_API_KEY)

# ========= SMALL IN-MEMORY SESSION =========
SESS = {}
SESS_LOCK = threading.Lock()
SESSION_TTL = 60 * 30  # 30 phút không tương tác thì reset

def _get_sess(user_id):
    now = time.time()
    with SESS_LOCK:
        s = SESS.get(user_id)
        if not s or now - s["ts"] > SESSION_TTL:
            s = {"hist": deque(maxlen=8), "last_mid": None, "ts": now}
            SESS[user_id] = s
        s["ts"] = now
        return s

def _remember(user_id, role, text):
    s = _get_sess(user_id)
    s["hist"].append({"role": role, "content": text})

# ========= OPENAI WRAPPER =========
def _to_chat_messages(messages):
    """Chuyển format responses -> chat.completions để fallback."""
    chat_msgs = []
    ALLOWED = {"input_text", "output_text", "text"}  # <-- thêm output_text
    for m in messages:
        role = m.get("role", "user")
        parts = m.get("content", [])
        text = "\n".join([p.get("text","") for p in parts if p.get("type") in ALLOWED]).strip()
        chat_msgs.append({"role": role, "content": text})
    return chat_msgs


def call_openai(messages, temperature=0.7):
    """
    Ưu tiên /v1/responses; nếu lỗi -> fallback /v1/chat/completions.
    messages: [{"role":..., "content":[{"type":"input_text","text":"..."}]}]
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "input": messages, "temperature": temperature}
    try:
        t0 = time.time()
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=40)
        dt = (time.time() - t0) * 1000
        print(f"🔁 OpenAI responses status={r.status_code} in {dt:.0f}ms")
        if r.status_code == 200:
            data = r.json()
            try:
                reply = data["output"][0]["content"][0]["text"]
            except Exception:
                reply = data.get("output_text") or "Mình đang ở đây, sẵn sàng hỗ trợ bạn!"
            return data, reply

        print(f"❌ responses body: {r.text[:800]}")
        # Fallback sang chat.completions
        chat_msgs = _to_chat_messages(messages)
        rc = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model": OPENAI_MODEL, "messages": chat_msgs, "temperature": temperature},
            timeout=40
        )
        print(f"🔁 OpenAI chat.status={rc.status_code}")
        if rc.status_code == 200:
            data = rc.json()
            reply = (data.get("choices") or [{}])[0].get("message", {}).get("content") or "..."
            return data, reply

        print(f"❌ chat body: {rc.text[:800]}")
        return {}, "Xin lỗi, hiện mình gặp chút trục trặc. Bạn nhắn lại giúp mình nhé!"
    except Exception as e:
        print("❌ OpenAI error:", repr(e))
        return {}, "Xin lỗi, hiện mình gặp chút trục trặc. Bạn nhắn lại giúp mình nhé!"

# === Rephrase mềm + emoji cute ===
EMOJI_SETS = {
    "generic": ["✨","🙂","😊","🌟","💫"],
    "greet":   ["👋","😊","🙂","✨"],
    "browse":  ["🛍️","🧭","🔎","✨"],
    "product": ["🛍️","✨","👍","💖"],
    "oos":     ["🙏","⛔","😅","🛒"],
    "policy":  ["ℹ️","📦","🛡️","✅"]
}
def em(intent="generic", n=1):
    if EMOJI_MODE == "none": return ""
    arr = EMOJI_SETS.get(intent, EMOJI_SETS["generic"])
    return " " + " ".join(random.choice(arr) for _ in range(max(1, n))).strip()

def rephrase_casual(text: str, intent="generic", temperature=0.7, lang: str = None) -> str:
    """Làm mềm câu + thêm 1–2 emoji nhẹ nhàng, đúng ngôn ngữ lang."""
    if not REPHRASE_ENABLED:
        return text + (em(intent,1) if intent!="generic" else "")
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        msgs = [
            {"role":"system","content":f"Bạn là trợ lý bán hàng, viết {lang or 'vi'} tự nhiên, thân thiện, ngắn gọn; thêm 1–2 emoji phù hợp (không lạm dụng). Giữ nguyên dữ kiện/giá, không bịa."},
            {"role":"user","content": f"Viết lại đoạn sau bằng {lang or 'vi'} theo giọng thân thiện, kết thúc bằng 1 câu chốt hành động.\n---\n{text}\n---\n{em(intent,2)}"}
        ]
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model": OPENAI_MODEL, "messages": msgs, "temperature": temperature, "max_tokens": 220},
            timeout=20
        )
        if r.status_code == 200:
            data = r.json()
            out = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            return out.strip() if out else text
        else:
            print("⚠️ rephrase status=", r.status_code, r.text[:200])
            return text + em(intent,1)
    except Exception as e:
        print("⚠️ rephrase error:", repr(e))
        return text + em(intent,1)
def handle_smalltalk(text: str, lang: str = "vi") -> str:
    # trả lời ngắn gọn, không gọi rephrase để tránh thêm CTA bán hàng
    raw = f"{t(lang, 'smalltalk_hi')} {t(lang, 'smalltalk_askback')}".strip()
    return raw


# ========= FACEBOOK SENDER =========
def fb_call(path, payload=None, method="POST", params=None, page_token=None):
    if not page_token:
        print("❌ missing page_token for fb_call")
        return None
    url = f"https://graph.facebook.com/v19.0{path}"
    params = params or {}
    params["access_token"] = page_token
    try:
        r = requests.request(method, url, params=params, json=payload, timeout=15)
        return r
    except Exception as e:
        print("⚠️ FB API error:", repr(e))
        return None

def fb_mark_seen(user_id, page_token):
    fb_call("/me/messages", {"recipient":{"id":user_id}, "sender_action":"mark_seen"}, page_token=page_token)

def fb_typing_on(user_id, page_token):
    fb_call("/me/messages", {"recipient":{"id":user_id}, "sender_action":"typing_on"}, page_token=page_token)

def fb_send_text(user_id, text, page_token):
    r = fb_call("/me/messages", {"recipient":{"id":user_id}, "message":{"text":text}}, page_token=page_token)
    print(f"📩 Send text status={getattr(r, 'status_code', None)}")

def fb_send_buttons(user_id, text, buttons, page_token):
    if not buttons: return
    payload = {
        "recipient": {"id": user_id},
        "message": {
            "attachment": {"type": "template","payload": {"template_type": "button","text": text,"buttons": buttons[:2]}}
        }
    }
    r = fb_call("/me/messages", payload, page_token=page_token)
    print(f"🔘 ButtonAPI status={getattr(r,'status_code',None)}")

# === Reload vectors (FAISS) ===
CANONICAL_DOMAIN = os.getenv("CANONICAL_DOMAIN", SHOP_URL).rstrip("/")
ALIAS_DOMAINS = [d.strip().rstrip("/") for d in os.getenv("ALIAS_DOMAINS","").split(",") if d.strip()]

def _canon_url(u: str) -> str:
    if not u: return u
    uu = u.strip()
    for dom in ALIAS_DOMAINS:
        if uu.startswith(dom + "/") or uu == dom:
            return CANONICAL_DOMAIN + uu[len(dom):]
    return uu

def _apply_canonical_urls(meta):
    if not meta: return meta
    for d in meta:
        if d.get("url"):
            d["url"] = _canon_url(d["url"])
    return meta

# ========= RAG (FAISS) =========

def _safe_read_index(prefix):
    try:
        idx_path  = os.path.join(VECTOR_DIR, f"{prefix}.index")
        meta_path = os.path.join(VECTOR_DIR, f"{prefix}.meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            print(f"⚠️ Missing index/meta for '{prefix}'")
            return None, None
        idx  = faiss.read_index(idx_path)
        meta = json.load(open(meta_path, encoding="utf-8"))

        # --- Áp canonical domain cho mọi URL trong meta ---
        meta = _apply_canonical_urls(meta)

        print(f"✅ {prefix} loaded: {len(meta)} chunks")
        return idx, meta
    except Exception as e:
        print(f"❌ Load index '{prefix}':", repr(e))
        return None, None
    
IDX_PROD, META_PROD = _safe_read_index("products")
IDX_POL,  META_POL  = _safe_read_index("policies")

def _reload_vectors():
    global IDX_PROD, META_PROD, IDX_POL, META_POL
    try:
        IDX_PROD, META_PROD = _safe_read_index("products")
        IDX_POL,  META_POL  = _safe_read_index("policies")
        ok = (IDX_PROD is not None or IDX_POL is not None)
        print("🔄 Reload vectors:", ok,
              "| prod_chunks=", (len(META_PROD) if META_PROD else 0),
              "| policy_chunks=", (len(META_POL) if META_POL else 0))
        return ok
    except Exception as e:
        print("❌ reload vectors:", repr(e))
        return False

@app.post("/admin/reload_vectors")
def admin_reload_vectors():
    ok = _reload_vectors()
    return jsonify({"ok": ok})

def _embed_query(q: str) -> np.ndarray:
    t0 = time.time()
    resp = oai_client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    print(f"🧩 Embedding in {(time.time()-t0)*1000:.0f}ms")
    return v

def search_products_with_scores(query, topk=8):
    if IDX_PROD is None:
        return [], []
    v = _embed_query(query)
    try:
        D, I = IDX_PROD.search(v, topk)
        hits, scores, seen = [], [], set()
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(META_PROD):
                d = META_PROD[idx]
                key = (d.get("url"), (d.get("title") or "").lower().strip())
                if key in seen:
                    continue
                seen.add(key)
                hits.append(d)
                scores.append(float(score))
        print(f"📚 product hits: {len(hits)}")
        return hits, scores
    except Exception as e:
        print("⚠️ search_products_with_scores:", repr(e))
        return [], []

def retrieve_context(question, topk=6):
    if IDX_PROD is None and IDX_POL is None:
        return ""
    v = _embed_query(question)
    ctx = []
    if IDX_PROD is not None:
        try:
            _, Ip = IDX_PROD.search(v, topk)
            ctx += [META_PROD[i]["text"] for i in Ip[0] if i >= 0]
        except Exception as e:
            print("⚠️ search products:", repr(e))
    if IDX_POL is not None:
        try:
            _, Ik = IDX_POL.search(v, topk)
            ctx += [META_POL[i]["text"] for i in Ik[0] if i >= 0]
        except Exception as e:
            print("⚠️ search policies:", repr(e))
    print("🧠 ctx pieces:", len(ctx))
    return "\n\n".join(ctx[:topk]) if ctx else ""
def _parse_ts(s):
    try:
        s = (s or "").replace("Z","").replace("T"," ")
        return time.mktime(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return 0

def get_new_arrivals(days=30, topk=4):
    """Tìm sp mới theo timestamp/tags 'new|mới|vừa về'; fallback FAISS nếu trống."""
    if not META_PROD:
        return []
    now = time.time()
    cutoff = now - days*86400
    new_items = []
    for d in META_PROD:
        ts = 0
        for k in ("created_at","published_at","updated_at"):
            if d.get(k):
                ts = max(ts, _parse_ts(d.get(k)))
        tags = (d.get("tags","") or "").lower()
        flag_new = any(x in tags for x in ["new","mới","vừa về","new arrivals"])
        if flag_new or (ts and ts >= cutoff):
            new_items.append(d)

    if not new_items and IDX_PROD is not None:
        hits, _ = search_products_with_scores("new arrivals hàng mới vừa về", topk=topk*2)
        new_items = hits

    def _key(d):
        ts = 0
        for k in ("created_at","published_at","updated_at"):
            ts = max(ts, _parse_ts(d.get(k)))
        return ts
    new_items.sort(key=_key, reverse=True)
    return new_items[:topk]

def compose_new_arrivals(lang: str = "vi", items=None):
    items = items or []
    if not items:
        url = SHOP_URL_MAP.get(lang, SHOP_URL_MAP.get(DEFAULT_LANG, SHOP_URL))
        return rephrase_casual(t(lang,"browse", url=url), intent="browse", lang=lang)
    lines = []
    for d in items[:2]:
        title = d.get("title") or "Sản phẩm"
        price = d.get("price")
        stock = _stock_line(d)
        line = f"• {title}"
        if price: line += f" — {price} đ"
        line += f" — {stock}"
        lines.append(line)
    raw = f"{t(lang,'new_hdr')}\n" + "\n".join(lines) + "\n\n" + t(lang,"product_pts")
    return rephrase_casual(raw, intent="product", lang=lang)


# ========= INTENT, PERSONA, FEW-SHOT & NATURAL REPLY =========
GREETS = {"hi","hello","hey","helo","heloo","hí","hì","chào","xin chào","alo","aloha","hello bot","hi bot"}
def is_greeting(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return any(w in t for w in GREETS) and len(t) <= 40

# ——— Ngôn ngữ: detect & câu chữ
def detect_lang(text: str) -> str:
    txt = (text or "").strip()
    if not txt: return DEFAULT_LANG
    if re.search(r"[\u4e00-\u9fff]", txt):  # CJK
        return "zh" if "zh" in SUPPORTED_LANGS else DEFAULT_LANG
    if re.search(r"[\u0E00-\u0E7F]", txt):  # Thai
        return "th" if "th" in SUPPORTED_LANGS else DEFAULT_LANG
    if re.search(r"[ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộơóờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵ]", txt, flags=re.I):
        return "vi" if "vi" in SUPPORTED_LANGS else DEFAULT_LANG
    if re.search(r"\b(yang|dan|tidak|saja|terima|kasih)\b", txt.lower()):
        return "id" if "id" in SUPPORTED_LANGS else DEFAULT_LANG
    return "en" if "en" in SUPPORTED_LANGS else DEFAULT_LANG


# ====== MULTI-LANG PATTERNS (smalltalk & new arrivals) ======
# Mỗi ngôn ngữ là 1 list regex. Có thể bổ sung dần mà không đụng chỗ khác.
# ==== Smalltalk & New arrivals (multi-lang) ====

# ========= I18N STRINGS & HELPERS =========
LANG_STRINGS = {
    "vi": {
        "greet": "Xin chào 👋 Rất vui được phục vụ bạn! Bạn muốn mình giúp gì không nè? 🙂",
        "browse": "Mời bạn vào web tham quan ạ 🛍️ 👉 {url}",
        "oos": "Xin lỗi 🙏 sản phẩm đó hiện **đang hết hàng** tại shop. Bạn thử xem các mẫu tương tự trên web nhé 👉 {url} ✨",
        "fallback": "Mình chưa đủ dữ liệu để chắc chắn 🤔. Bạn mô tả thêm mẫu/kiểu dáng/chất liệu để mình tư vấn chuẩn hơn nha ✨",
        "suggest_hdr": "Mình đề xuất vài lựa chọn phù hợp",
        "product_pts": "Bạn thích kiểu mảnh hay thể thao? Mình lọc thêm màu & size giúp bạn nhé.",
        "highlights": "{title} có vài điểm nổi bật nè",
        "policy_hint": "Theo chính sách shop:",
        "smalltalk_hi": "Hi 👋 Mình khỏe nè 😄",
        "smalltalk_askback": "Hôm nay của bạn thế nào?",
        "new_hdr": "Hàng mới về nè ✨",
    },
    "en": {
        "greet": "Hello 👋 Happy to help! How can I assist you today? 🙂",
        "browse": "Feel free to explore our store 🛍️ 👉 {url}",
        "oos": "Sorry 🙏 that item is **out of stock** right now. Check similar picks here 👉 {url} ✨",
        "fallback": "I’m missing a bit of info 🤔. Share style/material/size and I’ll refine the picks ✨",
        "suggest_hdr": "Here are a few good options",
        "product_pts": "Prefer a slim or sporty style? I can filter color & size for you.",
        "highlights": "{title} highlights",
        "policy_hint": "Store policy:",
         "smalltalk_hi": "Hi 👋 I'm good! 😄",
        "smalltalk_askback": "How's your day going?",
        "new_hdr": "New arrivals ✨",
    },
    "zh": {
        "greet": "你好 👋 很高兴为你服务！需要我帮你做什么呢？🙂",
        "browse": "欢迎逛逛我们的商店 🛍️ 👉 {url}",
        "oos": "抱歉 🙏 该商品目前**缺货**。可以先看看类似的款式 👉 {url} ✨",
        "fallback": "还需要一些信息哦 🤔。说下风格/材质/尺寸，我再精准推荐 ✨",
        "suggest_hdr": "给你几款合适的选择",
        "product_pts": "想要纤细还是运动风？我可以按颜色和尺码再筛一轮。",
        "highlights": "{title} 的亮点",
        "policy_hint": "店铺政策：",
        "smalltalk_hi": "嗨 👋 我很好喔 😄",
        "smalltalk_askback": "你今天过得怎么样？",
        "new_hdr": "新品上架 ✨",
    },
    "th": {
        "greet": "สวัสดี 👋 ยินดีให้บริการนะครับ/ค่ะ ต้องการให้ช่วยอะไรบ้าง 🙂",
        "browse": "เชิญชมสินค้าในเว็บได้เลย 🛍️ 👉 {url}",
        "oos": "ขออภัย 🙏 สินค้าชิ้นนั้น **หมดชั่วคราว** ค่ะ ลองดูรุ่นใกล้เคียงที่นี่ 👉 {url} ✨",
        "fallback": "ขอรายละเอียดเพิ่มอีกนิดนะคะ/ครับ เช่นสไตล์/วัสดุ/ขนาด ✨",
        "suggest_hdr": "ขอแนะนำตัวเลือกที่เหมาะสม",
        "product_pts": "ชอบแบบเพรียวหรือสปอร์ตดี? เดี๋ยวช่วยคัดสีและไซซ์ให้อีกได้ค่ะ/ครับ",
        "highlights": "จุดเด่นของ {title}",
        "policy_hint": "นโยบายร้าน:",
         "smalltalk_hi": "ไฮ 👋 สบายดีมากเลยนะ 😄",
        "smalltalk_askback": "วันนี้ของคุณเป็นยังไงบ้าง?",
        "new_hdr": "สินค้าเข้าใหม่ ✨",
    },
    "id": {
        "greet": "Halo 👋 Senang membantu! Ada yang bisa saya bantu? 🙂",
        "browse": "Silakan jelajahi toko kami 🛍️ 👉 {url}",
        "oos": "Maaf 🙏 produk itu **sedang kosong**. Coba lihat yang mirip di sini 👉 {url} ✨",
        "fallback": "Butuh info tambahan 🤔. Sebutkan gaya/bahan/ukuran ya, biar saya saringkan ✨",
        "suggest_hdr": "Beberapa pilihan yang cocok",
        "product_pts": "Suka model tipis atau sporty? Saya bisa saring warna & ukuran.",
        "highlights": "Hal menarik dari {title}",
        "policy_hint": "Kebijakan toko:",
       "smalltalk_hi": "Hai 👋 Aku baik-baik saja 😄",
        "smalltalk_askback": "Harinya kamu gimana?",
        "new_hdr": "Produk baru ✨",
    },
}


def t(lang: str, key: str, **kw) -> str:
    lang2 = lang if lang in LANG_STRINGS else DEFAULT_LANG
    s = (LANG_STRINGS.get(lang2, {}).get(key)
         or LANG_STRINGS.get(DEFAULT_LANG, {}).get(key)
         or "")
    try:
        return s.format(**kw)
    except Exception:
        return s


def greet_text(lang: str) -> str:
    return t(lang, "greet")

# ==== Smalltalk & New arrivals (multi-lang) ====

SMALLTALK_PATTERNS = {
    "vi": [
        # bn/bạn khỏe/khỏe/khoẻ không/ko/hông/hem/hok
        r"\b(bn|bạn)\s*(kh[oóòỏõọơớờởỡợ]e|khoe|khoẻ|khỏe)\s*(kh[oô]ng|ko|k|h[oơôóòõỏọ]ng|hong|hông|hem|hok)\b",
        r"\b(kh[oóòỏõọơớờởỡợ]e|khoe|khoẻ|khỏe)\s*(kh[oô]ng|ko|k|h[oơôóòõỏọ]ng|hong|hông|hem|hok)\b",
        # ổn không
        r"\b(ổn|on)\s*(kh[oô]ng|ko|k|hong|h[oơ]ng|hem|hok)\b",
        # hôm nay thế nào / nay sao
        r"\b(h[oô]m?\s*nay|nay)\s*(bạn|bn)?\s*(th[ếe]\s*n[aà]o|sao|ok\s*kh[oô]ng)\b",
        # đang làm gì / dạo này
        r"\b(đang\s*làm\s*gì|làm\s*gì( vậy| đó)?|làm\s*chi|lam\s*gi)\b",
        r"\b(dạo\s*này|dao\s*nay)\b",
        # ăn cơm chưa / ngủ chưa
        r"\b(ăn\s*cơm\s*chưa|ăn\s*chưa|an\s*chua|uống\s*chưa|ng[uư]\s*ch[aă]u?)\b",
        # cảm ơn / thanks
        r"\b(c[ảa]m\s?ơn|c[áa]m\s?ơn|thanks?|thank you|ty|tks|thx)\b",
        # cười/emoji
        r"\b(haha+|hihi+|hehe+|kkk+|=D|:d|:v|:3)\b|[😂🤣😆]",
    ],
    "en": [
        r"\b(how('?s)?\s*it\s*going|how\s*are\s*(you|u)|how\s*r\s*u|how\s*u\s*doin?g?)\b",
        r"\b(what('?s)?\s*up|wass?up|sup|wyd)\b",
        r"\b(have\s*you\s*eaten|had\s*(lunch|dinner)|grabbed\s*(lunch|food))\b",
        r"\b(thanks?|thank\s*(you|u)|ty|thx|tysm|tks)\b",
        r"\b(lol|lmao|rofl|haha+|hehe+|:d)\b|[😂🤣😆]",
    ],
    "zh": [
        r"(你好吗|妳好吗|最近怎么样|最近如何|最近还好|还好吗|心情如何|开心吗|過得怎樣|过得怎样)",
        r"(吃饭了吗|吃过饭没|吃了没|吃了吗)",
        r"(谢谢|多谢|謝謝|感謝|感谢|謝啦|谢谢啦|谢啦)",
        r"(哈哈+|嘿嘿+|呵呵+|嗨嗨+)|[😂🤣😆]",
    ],
    "th": [
        r"(สบายดี(ไหม|มั้ย|ป่าว)|เป็น(ไง|อย่างไร)บ้าง|โอเค(ไหม|มั้ย))",
        r"(ทำอะไรอยู่|กำลังทำอะไร|ทำไรอยู่)",
        r"(กินข้าว(หรือ)?ยัง|ทานข้าว(หรือ)?ยัง)",
        r"(ขอบคุณ(ครับ|ค่ะ)?|ขอบใจ|thanks?|thank you|ty)",
        r"(ฮ่า+ๆ+|555+)|[😂🤣😆]",
    ],
    "id": [
        r"(apa\s*kabar|gimana\s*kabarnya|gmn\s*kabar|kabarnya\s*gimana)",
        r"(lagi\s*apa|lg\s*apa|sedang\s*apa|ngapain(\s*nih)?)",
        r"(sudah|udah)\s*makan\s*(belum|blm)",
        r"(terima\s*kasih|terimakasih|trimakasih|makasih|makasi|thanks?|thank you|thx|ty)",
        r"(wkwk+|wk+|haha+|hehe+|:d)|[😂🤣😆]",
    ],
}

NEW_ITEMS_PATTERNS = {
    "vi": [
        r"(hàng|sp|mẫu|sản\s*phẩm).*(mới|vừa\s*về|new\s*arrivals)",
        r"(có|đã).*(mẫu|sản\s*phẩm).*(mới|vừa\s*về)",
        r"\b(new|mới|vừa về|new arrivals)\b",
    ],
    "en": [
        r"(new\s*arrivals?|new\s*products?|what's\s*new)",
        r"(any|have).*(new\s*items?)",
    ],
    "zh": [
        r"(新品|新到|新貨|新货)",
        r"(有.*新(品|货|貨)|來了.*新|来了.*新)",
    ],
    "th": [
        r"(สินค้าเข้าใหม่|ของเข้าใหม่|ของใหม่|มาใหม่)",
        r"(มีอะไรใหม่|มีสินค้าใหม่ไหม)",
    ],
    "id": [
        r"(produk baru|barang baru|baru datang)",
        r"(ada yang baru|ada produk baru)",
    ],
}

def _pat(pats: dict, lang: str):
    """Lấy list pattern theo ngôn ngữ, fallback về DEFAULT_LANG nếu không có."""
    return pats.get(lang) or pats.get(DEFAULT_LANG, [])
# ===== Giá / Price questions (multi-lang) =====
PRICE_PATTERNS = {
    "vi": [r"\bgiá\b", r"bao nhiêu", r"nhiêu tiền", r"\bgiá bao nhiêu\b", r"\bbao nhieu\b"],
    "en": [r"\bprice\b", r"how much", r"\bcost\b"],
    "zh": [r"(价格|幾錢|多少钱|多少錢)"],
    "th": [r"(ราคา|เท่าไหร่|เท่าไร)"],
    "id": [r"(harga|berapa)"],
}
def is_price_question(text: str, lang: str) -> bool:
    raw = (text or "")
    return any(re.search(p, raw, flags=re.I) for p in _pat(PRICE_PATTERNS, lang))


SYSTEM_STYLE = (
    "Bạn là trợ lý bán hàng Aloha tên là Aloha Bot. Tông giọng: thân thiện, chủ động, "
    "trả lời ngắn gọn như người thật; dùng 1–3 emoji hợp ngữ cảnh (không lạm dụng). "
    "Luôn dựa vào CONTEXT (nội dung RAG). Không bịa. Nếu thiếu dữ liệu thực tế, nói 'mình chưa có dữ liệu' "
    "và hỏi lại 1 câu để làm rõ. Trình bày dễ đọc: gạch đầu dòng khi liệt kê; 1 câu chốt hành động."
)

# FEW_SHOT_EXAMPLES
FEW_SHOT_EXAMPLES = [
    {"role":"user","content":[{"type":"input_text","text":"helo"}]},
    {"role":"assistant","content":[{"type":"output_text","text":"Xin chào 👋 Rất vui được phục vụ bạn! Bạn muốn mình giúp gì không nè? 🙂"}]},
    {"role":"user","content":[{"type":"input_text","text":"shop bạn có những gì"}]},
    {"role":"assistant","content":[{"type":"output_text","text":f"Mời bạn tham quan cửa hàng tại đây ạ 🛍️ 👉 {SHOP_URL_MAP.get('vi', SHOP_URL)}"}]},
]
# ---- Intent routing ----
POLICY_KEYWORDS  = {"chính sách","đổi trả","bảo hành","ship","vận chuyển","giao hàng","trả hàng","refund"}
PRODUCT_KEYWORDS = {
    "mua","bán","giá","size","kích thước","chất liệu","màu","hợp","phù hợp",
    "dây","đồng hồ","vòng","case","áo","quần","áo phông","tshirt","t-shirt","áo thun",
    "sản phẩm", "bánh","crepe","bánh crepe","bánh sầu riêng","milktea","trà sữa"
}
BROWSE_KEYWORDS  = {"có những gì","bán gì","có gì","danh mục","catalog","xem hàng","tham quan","xem shop","xem sản phẩm","shop có gì","những sản phẩm gì"}
_BROWSE_PATTERNS = [
    r"(shop|bên bạn|bên mình).*(bán|có).*(gì|những gì|những sản phẩm gì)",
    r"(bán|có).*(những\s+)?sản phẩm gì",
]
# ==== Smalltalk & New arrivals ====



def detect_intent(text: str):
    raw = (text or "")
    t0  = re.sub(r"\s+", " ", raw.lower()).strip()
    lang = detect_lang(raw)

    if any(k in t0 for k in POLICY_KEYWORDS):  return "policy"
    if is_greeting(raw):                       return "greet"

    # smalltalk đa ngôn ngữ
    if any(re.search(p, raw, flags=re.I) for p in _pat(SMALLTALK_PATTERNS, lang)):
        return "smalltalk"

    # browse: từ khóa + pattern chung
    if any(k in t0 for k in BROWSE_KEYWORDS):  return "browse"
    if any(re.search(p, t0) for p in _BROWSE_PATTERNS):     return "browse"

    # hỏi hàng mới đa ngôn ngữ
    if any(re.search(p, raw, flags=re.I) for p in _pat(NEW_ITEMS_PATTERNS, lang)):
        return "new_items"
    # Hỏi giá → ưu tiên product_info
    if is_price_question(raw, lang):
        return "product_info"

    # sản phẩm & mô tả
    if any(k in t0 for k in PRODUCT_KEYWORDS): return "product"
    if "có bán" in t0 or "bán không" in t0 or "bán ko" in t0: return "product"
    if "có gì đặc biệt" in t0 or "điểm đặc biệt" in t0 or "có gì đặt biệt" in t0: return "product_info"

    return "other"

def build_messages(system, history, context, user_question):
    msgs = [{"role":"system","content":[{"type":"input_text","text":system}]}]
    msgs.extend(FEW_SHOT_EXAMPLES)
    for h in list(history)[-3:]:
        ctype = "output_text" if h["role"] == "assistant" else "input_text"
        msgs.append({"role": h["role"], "content":[{"type": ctype, "text": h["content"]}]})
    user_block = f"(Nếu hữu ích thì dùng CONTEXT)\nCONTEXT:\n{context}\n\nCÂU HỎI: {user_question}"
    msgs.append({"role":"user","content":[{"type":"input_text","text":user_block}]})
    return msgs


# ---- Hiển thị tồn kho/OOS + emoji ----
def _stock_line(d: dict) -> str:
    if d.get("available") and (d.get("inventory_quantity") is None or d.get("inventory_quantity", 0) > 0):
        return "còn hàng ✅"
    if d.get("inventory_quantity") == 0 or (d.get("status") and d.get("status") != "active"):
        return "hết hàng tạm thời ⛔"
    return "tình trạng đang cập nhật ⏳"

def _shorten(txt: str, n=280) -> str:
    t = (txt or "").strip()
    return (t[:n].rstrip() + "…") if len(t) > n else t
def _fmt_price(p, currency="₫"):
    if p is None:
        return None
    try:
        s = re.sub(r"[^\d.]", "", str(p))
        if not s:
            return None
        val = int(float(s))
        return f"{val:,.0f}".replace(",", ".") + (f" {currency}" if currency else "")
    except Exception:
        return str(p)

def _extract_price_number(txt: str):
    """Trả về số (float) nếu bắt được 199k/199.000đ/199000 vnd..., else None"""
    if not txt:
        return None
    low = txt.lower()
    m = re.search(r"(\d[\d\.\s,]{2,})(?:\s?)(đ|₫|vnd|vnđ|k)\b", low)
    try:
        if m:
            num = re.sub(r"[^\d.]", "", m.group(1))
            v = float(num)
            return v*1000 if m.group(2) == "k" else v
        m2 = re.search(r"\b(\d{5,})\b", low)
        return float(m2.group(1)) if m2 else None
    except Exception:
        return None

def _price_value(d: dict):
    """Trả numeric price tốt nhất từ meta; fallback bắt trong text."""
    for k in ("price","min_price","max_price"):
        v = d.get(k)
        if v is not None:
            try:
                return float(re.sub(r"[^\d.]", "", str(v)))
            except Exception:
                pass
    return _extract_price_number(d.get("text",""))

def _category_key_from_doc(d: dict):
    """Xác định 'dòng' sản phẩm để so min–max: ưu tiên product_type; fallback theo synonyms trong title/tags."""
    pt = (d.get("product_type") or "").strip()
    if pt:
        return _normalize_text(pt)
    raw = " ".join([d.get("title",""), d.get("tags","")])
    n1, n2 = _norm_both(raw)
    for key in VN_SYNONYMS.keys():
        k1, k2 = _norm_both(key)
        if k1 in n1 or k2 in n2:
            return _normalize_text(key)
    return "misc"

def _minmax_in_category(base_doc: dict):
    """Tìm 1 mẫu rẻ nhất và 1 mẫu đắt nhất cùng dòng (loại). Nếu không đủ, fallback toàn shop."""
    if not META_PROD:
        return None, None
    cat = _category_key_from_doc(base_doc)
    def same_cat(x):
        return _category_key_from_doc(x) == cat
    cands = [x for x in META_PROD if same_cat(x)]
    if len(cands) < 2:
        cands = [x for x in META_PROD]  # fallback toàn shop

    items = []
    for x in cands:
        pv = _price_value(x)
        if pv is not None:
            items.append((pv, x))
    if not items:
        return None, None

    # loại chính ra khỏi candidates nếu trùng URL hoặc title
    def is_same(a, b):
        return (a.get("url") and a.get("url")==b.get("url")) or \
               ((a.get("title") or "").strip().lower()==(b.get("title") or "").strip().lower())

    items = [(p, x) for (p, x) in items if not is_same(x, base_doc)]
    if not items:
        return None, None

    items.sort(key=lambda t: t[0])
    low = items[0][1]
    high = items[-1][1]
    return low, high


def _extract_features_from_text(text_block: str):
    lines = []
    m = re.search(r"Specs:\s*(.+)", text_block, flags=re.I)
    if m:
        lines.extend(re.split(r"\s*\|\s*|\s*;\s*|,\s*", m.group(1)))
    m2 = re.search(r"Body:\s*(.+?)\s*URL:", text_block, flags=re.I|re.S)
    if m2:
        body = re.sub(r"\s+", " ", m2.group(1)).strip()
        chunks = re.split(r"(?<=[.!?])\s+", body)
        lines.extend(chunks)
    lines = [re.sub(r"\s+", " ", l).strip("•- \n\t") for l in lines if l and len(l.strip()) > 0]
    uniq = []
    for l in lines:
        if l not in uniq:
            uniq.append(l)
        if len(uniq) >= 5:
            break
    return ["• " + _shorten(x, 80) for x in uniq[:5]]

# ====== TỪ KHÓA / ĐỒNG NGHĨA (đa ngôn ngữ) ======
VN_SYNONYMS = {
    # ===== Đồng hồ & phụ kiện =====
    "đồng hồ": [
        "dong ho","dong-ho","dongho","watch","watchface","watch face","bezel",
        "galaxy watch","apple watch","amazfit","seiko","casio","nh35","nh36",
        "automatic","mechanical","chronograph"
    ],
    "dây đồng hồ": [
        "day dong ho","daydongho","watch band","band","strap","nato","loop",
        "bracelet","mesh","leather strap","metal strap","silicone strap"
    ],
    "case đồng hồ": [
        "case dong ho","vo dong ho","bao ve dong ho","bezel protector",
        "watch case","watch bumper","watch cover","protective case"
    ],
    "kính cường lực": [
        "kinh cuong luc","tempered glass","screen protector","glass protector",
        "full glass","full cover","full glue","9h","anti-scratch","privacy glass"
    ],
    "ốp lưng": [
        "op lung","case","cover","bumper","clear case","tpu case",
        "silicone case","shockproof case","phone case","protective case"
    ],
    "vòng tay": [
        "vong tay","bracelet","bangle","chain bracelet","cuff"
    ],
    "áo thun": [
        "ao thun","ao phong","tshirt","t-shirt","tee","tee shirt","crewneck",
        "basic tee","unisex tee","oversize tee"
    ],
    "áo phông": [
        "ao phong","ao thun","tshirt","t-shirt","tee"
    ],

    # ===== Đồ ngọt/đồ uống (bổ sung cho shop) =====
    "bánh": [
        "banh","cake","gateau","pastry","甜品","點心","เค้ก","ขนม",
        "kue","kueh","roti manis"
    ],
    "bánh crepe": [
        "banh crepe","crepe","mille crepe","crepe cake",
        "可丽饼","可麗餅","千层","千層","千层蛋糕","千層蛋糕",
        "เครป","เครปเค้ก","kue crepe","mille crepes","kue lapis"
    ],
    "bánh sầu riêng": [
        "banh sau rieng","durian","durian cake","durian crepe",
        "榴莲","榴槤","榴莲千层","榴槤千層","榴莲千层蛋糕","榴槤千層蛋糕","榴莲可丽饼","榴槤可麗餅",
        "เครปทุเรียน","เค้กทุเรียน",
        "kue durian","crepe durian","kue lapis durian"
    ],
    "trà sữa": [
        "tra sua","milk tea","bubble tea","boba","boba tea","pearl milk tea",
        "奶茶","珍珠奶茶","波霸奶茶",
        "ชานม","ชานมไข่มุก",
        "teh susu","teh susu boba","minuman boba","bubble tea id"
    ],
    "milktea": [
        "milk tea","bubble tea","boba","pearl milk tea",
        "奶茶","珍珠奶茶","ชานมไข่มุก","teh susu","boba tea"
    ],

    # ===== Chinese (ZH) – nhóm theo khái niệm để bắt rộng hơn =====
    "手表": ["腕表","watch","表带","表鏈","表圈","表壳","表殼","钢化膜","鋼化膜","保护壳","保護殼"],
    "表带": ["表帶","表链","表鏈","watch band","strap","皮表带","金属表带","硅胶表带"],
    "钢化膜": ["鋼化膜","玻璃膜","贴膜","貼膜","保护膜","保護膜","tempered glass","screen protector","全胶","全膠","9h"],
    "手机壳": ["手機殼","保护壳","保護殼","手机套","phone case","case","bumper","保護殼"],
    "T恤": ["T恤衫","短袖","圆领","圓領","tee","tshirt","t-shirt"], 
    "奶茶": ["珍珠奶茶","波霸奶茶","奶盖茶","milk tea","bubble tea","boba"],
    "可丽饼": ["可麗餅","法式薄饼","法式薄餅","千层","千層","千层蛋糕","千層蛋糕","crepe","mille crepe"],
    "榴莲": ["榴槤","durian","榴莲千层","榴槤千層","榴莲可丽饼","榴槤可麗餅","榴莲蛋糕","榴槤蛋糕"],

    # ===== Thai (TH) =====
    "นาฬิกา": ["watch","สายนาฬิกา","ฟิล์มกระจก","กรอบนาฬิกา","เคสนาฬิกา"],
    "สายนาฬิกา": ["watch band","strap","สายหนัง","สายโลหะ","สายซิลิโคน","นาโต้"],
    "ฟิล์มกระจก": ["tempered glass","กระจกกันรอย","full glue","9h","screen protector"],
    "เคสโทรศัพท์": ["เคส","ซองมือถือ","bumper","phone case","protective case"],
    "เสื้อยืด": ["tshirt","t-shirt","tee","คอกลม","โอเวอร์ไซซ์"],
    "ชานมไข่มุก": ["ชานม","bubble tea","boba","milk tea"],
    "เครป": ["เครปเค้ก","crepe","mille crepe"],
    "ทุเรียน": ["durian","เครปทุเรียน","เค้กทุเรียน"],

    # ===== Indonesian (ID) =====
    "jam tangan": ["watch","tali jam","pelindung layar","casing jam","bezel","case jam"],
    "tali jam": ["watch band","strap","nato","kulit","logam","silikon"],
    "pelindung layar": ["tempered glass","screen protector","kaca tempered","9h","full glue"],
    "casing hp": ["case","casing","sarung hp","bumper","phone case"],
    "kaos": ["tshirt","t-shirt","tee","kaos oversize","kaos unisex"],
    "teh susu": ["bubble tea","boba","milk tea","minuman boba"],
    "crepe": ["kue crepe","mille crepe","kue lapis","crepe cake"],
    "durian": ["kue durian","crepe durian","kue lapis durian"]
}



def _query_tokens(q: str, lang: str = "vi") -> set:
    """Sinh token từ câu hỏi: có dấu, không dấu, bigram, và cụm đồng nghĩa cơ bản."""
    n1, n2 = _norm_both(q)
    w1 = [w for w in n1.split() if len(w) > 1]
    w2 = [w for w in n2.split() if len(w) > 1]

    tokens = set(w1) | set(w2)

    # bigram cho cả có dấu & không dấu (để bắt “sầu riêng”, “banh sau”)
    for words in (w1, w2):
        for i in range(len(words) - 1):
            tokens.add((words[i] + " " + words[i+1]).strip())
            tokens.add((words[i] + words[i+1]).strip())  # biến thể không space

    combo_phrases = {
        "vi": ["đồng hồ","dây đồng hồ","kính cường lực","ốp lưng","áo thun","áo phông","bánh crepe","bánh sầu riêng","trà sữa"],
        "en": ["watch band","screen protector","phone case","t shirt","t-shirt","mille crepe","durian crepe","milk tea","bubble tea","boba tea"],
        "zh": ["手表","表带","钢化膜","手机壳","T恤","可丽饼","榴莲千层","奶茶","珍珠奶茶"],
        "th": ["นาฬิกา","สายนาฬิกา","ฟิล์มกระจก","เคสโทรศัพท์","เสื้อยืด","เครป","เครปทุเรียน","ชานมไข่มุก"],
        "id": ["jam tangan","tali jam","pelindung layar","casing hp","kaos","kue crepe","crepe durian","teh susu","bubble tea","boba"]
    }

    joined_n1 = " ".join(w1)
    for phrase in combo_phrases.get(lang, []):
        if phrase in joined_n1:
            tokens.add(_normalize_text(phrase))
            tokens.add(_normalize_text(_strip_accents(phrase)))

    # ánh xạ synonyms: nếu text chứa “key” thì thêm tất cả synonym vào tokens
    for key, syns in VN_SYNONYMS.items():
        key_n1, key_n2 = _norm_both(key)
        if key_n1 in n1 or key_n2 in n2:
            for s in syns:
                s1, s2 = _norm_both(s)
                tokens.add(s1)
                tokens.add(s2)
                tokens.add(s1.replace(" ", ""))
                tokens.add(s2.replace(" ", ""))

    return set(t for t in tokens if len(t) >= 2)


def filter_hits_by_query(hits, q, lang="vi"):
    """Giữ hit nếu có token/cụm từ câu hỏi xuất hiện trong title/tags/type/variant (có & không dấu)."""
    if not hits:
        return []
    qtoks = _query_tokens(q, lang=lang)

    kept = []
    for d in hits:
        hay_raw = " ".join([
            d.get("title",""), d.get("tags",""), d.get("product_type",""), d.get("variant","")
        ])
        h1, h2 = _norm_both(hay_raw)
        h1_ns, h2_ns = h1.replace(" ", ""), h2.replace(" ", "")

        ok = any(
            (t in h1) or (t in h2) or (t.replace(" ","") in h1_ns) or (t.replace(" ","") in h2_ns)
            for t in qtoks
        )
        if ok:
            kept.append(d)
    return kept


def should_relax_filter(q: str, hits: list) -> bool:
    qn = _normalize_text(q)
    return len(qn.split()) <= 2 and len(hits) > 0

# ====== Compose trả lời ======
def compose_product_reply(hits, lang: str = "vi"):
    if not hits:
        return t(lang, "fallback")

    # Ưu tiên currency trong meta; nếu không có, mặc định ₫ cho VI
    currency = (hits[0].get("currency") or ("₫" if lang == "vi" else ""))

    items = []
    for d in hits[:2]:
        title     = d.get("title") or "Sản phẩm"
        variant   = d.get("variant")
        stock     = _stock_line(d)

        price_val = _price_value(d)
        price_str = _fmt_price(price_val, currency) if price_val is not None else None

        line = f"• {title}"
        if variant:
            line += f" ({variant})"
        if price_str:
            line += f" — {price_str}"
        line += f" — {stock}"
        items.append(line)

    raw = f"{t(lang,'suggest_hdr')}\n" + "\n".join(items) + "\n\n" + t(lang,"product_pts")
    return rephrase_casual(raw, intent="product", lang=lang)

def compose_product_info(hits, lang: str = "vi"):
    if not hits:
        return t(lang, "fallback")

    d = hits[0]
    currency   = d.get("currency") or ("₫" if lang == "vi" else "")
    title      = d.get("title") or "Sản phẩm"
    stock      = _stock_line(d)

    price_val  = _price_value(d)
    price_line = f"Giá tham khảo: {_fmt_price(price_val, currency)}" if price_val is not None else ""

    bullets = _extract_features_from_text(d.get("text",""))
    body    = "\n".join(bullets) if bullets else "• Thiết kế tối giản, dễ phối đồ\n• Chất liệu thoáng, dễ vệ sinh"

    parts = [
        f"{t(lang,'highlights', title=title)}",
        body
    ]
    if price_line:
        parts.append(price_line)
    parts.extend([
        f"Tình trạng: {stock}",
        t(lang,"product_pts")
    ])

    raw = "\n".join(parts).strip()
    return rephrase_casual(raw, intent="product", lang=lang)


def compose_contextual_answer(context, question, history):
    msgs = build_messages(SYSTEM_STYLE, history, context, question)
    _, reply = call_openai(msgs, temperature=0.6)
    return reply

def compose_price_with_suggestions(hits, lang: str = "vi"):
    if not hits:
        return t(lang, "fallback"), []

    main = hits[0]
    currency = main.get("currency") or ("₫" if lang == "vi" else "")
    main_price = _price_value(main)
    main_price_str = _fmt_price(main_price, currency) if main_price is not None else "đang cập nhật"

    low, high = _minmax_in_category(main)

    lines = []
    title = main.get("title") or "Sản phẩm"
    lines.append(f"Vâng ạ, **{title}** đang được shop bán với **giá công khai: {main_price_str}**.")
    sug = []
    if high:
        hp = _fmt_price(_price_value(high), currency)
        sug.append(f"• **Cùng dòng – giá cao nhất:** {high.get('title','SP')} — {hp}")
    if low:
        lp = _fmt_price(_price_value(low), currency)  # đã sửa _ue → _price_value
        sug.append(f"• **Cùng dòng – giá thấp nhất:** {low.get('title','SP')} — {lp}")

    if sug:
        lines.append("Bạn cũng có thể tham khảo thêm:")
        lines += sug
    lines.append(t(lang, "product_pts"))
    raw = "\n".join(lines)

    btns = [x for x in (high, low) if x]
    return rephrase_casual(raw, intent="product", lang=lang), btns[:2]


def answer_with_rag(user_id, user_question):
    s = _get_sess(user_id)
    hist = s["hist"]

    intent = detect_intent(user_question)
    lang = detect_lang(user_question)
    print(f"🔎 intent={intent} | 🗣️ lang={lang}")

    # ——— QUICK ROUTES ———
    if intent == "greet":
        return greet_text(lang), []
    if intent == "smalltalk":
        return handle_smalltalk(user_question, lang=lang), []
    if intent == "browse":
        url = SHOP_URL_MAP.get(lang, SHOP_URL_MAP.get(DEFAULT_LANG, SHOP_URL))
        return t(lang, "browse", url=url), []
    if intent == "new_items":
        items = get_new_arrivals(days=30, topk=4)
        return compose_new_arrivals(lang=lang, items=items), items[:2]

    # ——— PRODUCT SEARCH ———
    prod_hits, prod_scores = search_products_with_scores(user_question, topk=8)
    best = max(prod_scores or [0.0])

    filtered_hits = filter_hits_by_query(prod_hits, user_question, lang=lang) if STRICT_MATCH else prod_hits
    if STRICT_MATCH and not filtered_hits and should_relax_filter(user_question, prod_hits):
        print("🔧 relaxed_filter=True (fallback to unfiltered hits)")
        filtered_hits = prod_hits

    # So khớp tiêu đề tính trên TOÀN BỘ prod_hits
    title_ok = _has_title_overlap(user_question, prod_hits)

    # Nếu gần đúng tiêu đề hoặc có hit → coi như intent=product
    if intent == "other" and (filtered_hits or title_ok):
        intent = "product"

    # Nếu trùng tiêu đề nhưng filtered rỗng → dùng lại prod_hits
    if title_ok and not filtered_hits:
        filtered_hits = prod_hits

    print(f"📈 best_score={best:.3f}, hits={len(prod_hits)}, kept_after_filter={len(filtered_hits)}, title_ok={title_ok}")

    # ——— CONTEXT/POLICY ———
    context = retrieve_context(user_question, topk=6)
    if intent == "policy" and context:
        ans = compose_contextual_answer(context, user_question, hist)
        ans = f"{t(lang,'policy_hint')} {ans}"
        return rephrase_casual(ans, intent="policy", temperature=0.5, lang=lang), []

    # ——— ƯU TIÊN HỎI GIÁ ———
    if is_price_question(user_question, lang) and (filtered_hits or title_ok):
        print("➡️ route=price_question→price_with_suggestions")
        chosen = filtered_hits if filtered_hits else prod_hits
        reply, sug_hits = compose_price_with_suggestions(chosen, lang=lang)
        return reply, sug_hits

    # ——— PRODUCT BRANCHES ———
    if intent in {"product", "product_info"}:
        # Không có hit hoặc score thấp & không trùng tiêu đề → OOS/fallback link
        if not filtered_hits or (best < SCORE_MIN and not title_ok):
            url = SHOP_URL_MAP.get(lang, SHOP_URL_MAP.get(DEFAULT_LANG, SHOP_URL))
            print("➡️ route=oos_hint")
            return t(lang, "oos", url=url), []

    if intent == "product_info":
        print("➡️ route=product_info")
        return compose_product_info(filtered_hits, lang=lang), filtered_hits[:1]

    if intent in {"product", "other"} and filtered_hits and (best >= SCORE_MIN or title_ok):
        print("➡️ route=product_reply")
        return compose_product_reply(filtered_hits, lang=lang), filtered_hits[:2]

    # ——— CONTEXT FALLBACK ———
    if context:
        ans = compose_contextual_answer(context, user_question, hist)
        print("➡️ route=ctx_fallback")
        return rephrase_casual(ans, intent="generic", temperature=0.7, lang=lang), []

    print("➡️ route=fallback")
    return t(lang, "fallback"), []


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        return (challenge, 200) if token == VERIFY_TOKEN else ("Invalid verification token", 403)

    payload = request.json or {}

    for entry in payload.get("entry", []):
        owner_id = str(entry.get("id"))           # Page ID hoặc IG Account ID
        access_token = TOKEN_MAP.get(owner_id)
        if not access_token:
            print("⚠️ No token mapped for:", owner_id); 
            continue

        for event in entry.get("messaging", []):
            if event.get("message", {}).get("is_echo"):
                continue
            if event.get("message") and "text" in event["message"]:
                psid = event["sender"]["id"]
                text = event["message"]["text"]
                mid  = event["message"].get("mid")

                sess = _get_sess(psid)
                if mid and sess["last_mid"] == mid: 
                    continue
                sess["last_mid"] = mid

                fb_mark_seen(psid, access_token)
                fb_typing_on(psid, access_token)

                _remember(psid, "user", text)
                reply, btn_hits = answer_with_rag(psid, text)
                _remember(psid, "assistant", reply)

                fb_send_text(psid, reply, access_token)

                if btn_hits:
                    buttons = []
                    for h in btn_hits[:2]:
                        if h.get("url"):
                            buttons.append({"type":"web_url","url":h["url"],"title":(h.get("title") or "Xem sản phẩm")[:20]})
                    if buttons:
                        fb_send_buttons(psid, "Xem nhanh:", buttons, access_token)

            elif event.get("postback", {}).get("payload"):
                psid = event["sender"]["id"]
                fb_send_text(psid, f"Bạn vừa chọn: {event['postback']['payload']}", access_token)
            # quick reply payload (nếu dùng quick_replies)
            elif event.get("message", {}).get("quick_reply", {}).get("payload"):
                psid = event["sender"]["id"]
                qr_payload = event["message"]["quick_reply"]["payload"]
                fb_send_text(psid, f"Bạn vừa chọn: {qr_payload}", access_token)


    return "ok", 200


# ========= API =========
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    print("🌐 /api/chat len =", len(messages))
    openai_raw, _ = call_openai(messages)
    return jsonify(openai_raw)

@app.route("/api/chat_rag", methods=["POST"])
def chat_rag():
    data = request.json or {}
    q = data.get("question", "")
    print("🌐 /api/chat_rag question:", q)
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400
    reply, _ = answer_with_rag("anonymous", q)
    return jsonify({"reply": reply})

@app.route("/api/product_search")
def api_product_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"ok": False, "msg": "missing q"}), 400
    lang = detect_lang(q)
    hits, scores = search_products_with_scores(q, topk=8)
    best = max(scores or [0.0])
    kept = filter_hits_by_query(hits, q, lang=lang) if STRICT_MATCH else hits
    if STRICT_MATCH and not kept and should_relax_filter(q, hits):
        kept = hits
    if not kept or best < SCORE_MIN:
        url = SHOP_URL_MAP.get(lang, SHOP_URL_MAP.get(DEFAULT_LANG, SHOP_URL))
        return jsonify({"ok": True, "reply": t(lang, "oos", url=url), "items": []})
    reply = compose_product_reply(kept, lang=lang)
    return jsonify({"ok": True, "reply": reply, "items": kept[:2]})

# ========= IG OAuth callback & policy pages =========
@app.route("/auth/callback")
def auth_callback():
    code = request.args.get("code")
    print("🔁 /auth/callback code:", code)
    return f"Auth success! Code: {code}"

@app.route("/privacy")
def privacy():
    return """
    <h1>Privacy Policy - Aloha Bot</h1>
    <p>Chúng tôi chỉ xử lý nội dung tin nhắn mà người dùng gửi tới Fanpage để trả lời.
    Không bán/chia sẻ dữ liệu cá nhân. Dữ liệu phiên trò chuyện (session) chỉ lưu tạm thời
    tối đa 30 phút phục vụ trả lời và sẽ tự xoá sau đó. Chỉ số sản phẩm (vectors) là dữ liệu công khai từ cửa hàng.</p>
    <p>Liên hệ xoá dữ liệu: gửi tin nhắn 'delete my data' tới Fanpage hoặc email: <b>hoclac1225@email.com</b>.</p>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/data_deletion")
def data_deletion():
    return """
    <h1>Data Deletion Instructions</h1>
    <p>Để yêu cầu xoá dữ liệu: (1) nhắn 'delete my data' tới Fanpage, hoặc (2) gửi email tới <b>hoclac1225@email.com</b>
    kèm ID cuộc trò chuyện. Chúng tôi sẽ xử lý trong thời gian sớm nhất.</p>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}

# ========= Debug & Health =========
@app.route("/debug/rag_status")
def rag_status():
    return jsonify({
        "vector_dir": os.path.abspath(VECTOR_DIR),
        "products_index": bool(IDX_PROD),
        "products_chunks": len(META_PROD) if META_PROD else 0,
        "policies_index": bool(IDX_POL),
        "policies_chunks": len(META_POL) if META_POL else 0,
        "sessions": len(SESS),
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "products": len(META_PROD) if META_PROD else 0,
        "policies": len(META_POL) if META_POL else 0
    })

# ========= Watcher: tự reload khi vector đổi =========
from apscheduler.schedulers.background import BackgroundScheduler

_last_vec_mtime = 0
def _mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0

def _watch_vectors():
    global _last_vec_mtime
    idx_p = os.path.join(VECTOR_DIR, "products.index")
    meta_p = os.path.join(VECTOR_DIR, "products.meta.json")
    idx_k = os.path.join(VECTOR_DIR, "policies.index")
    meta_k = os.path.join(VECTOR_DIR, "policies.meta.json")

    newest = max(_mtime(idx_p), _mtime(meta_p), _mtime(idx_k), _mtime(meta_k))
    if newest and newest != _last_vec_mtime:
        print("🕵️ Detected vector change → reload")
        if _reload_vectors():
            _last_vec_mtime = newest

def _start_vector_watcher():
    try:
        sch = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")
        sch.add_job(_watch_vectors, "interval", seconds=30, id="watch_vectors")
        sch.start()
        print("⏱️ Vector watcher started (30s)")
    except Exception as e:
        print("⚠️ Scheduler error:", repr(e))

# ======== MAIN ========
if __name__ == "__main__":
    _start_vector_watcher()
    port = int(os.getenv("PORT", 3000))
    print(f"🚀 Starting app on 0.0.0.0:{port}")
    # app.run(host="0.0.0.0", port=port, debug=False)  # khi chạy local
