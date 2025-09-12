# -*- coding: utf-8 -*-
import unicodedata
import os, json, time, re, requests, numpy as np, faiss, threading, random
from collections import deque
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import hmac, hashlib, base64


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
SCORE_MIN    = float(os.getenv("PRODUCT_SCORE_MIN", "0.30"))
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
    for m in messages:
        role = m.get("role", "user")
        parts = m.get("content", [])
        text = "\n".join([p.get("text","") for p in parts if p.get("type") in ("input_text","text")]).strip()
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
    alt = {
        "vi": ["Hôm nay shop nhiều năng lượng lắm nè ⚡", "Vui ghê, đang ship đơn đều tay 🤝"],
        "en": ["We’re full of energy today ⚡", "Orders are shipping steadily 🤝"],
        "zh": ["今天精神满满 ⚡", "订单正在稳定发货中 🤝"],
        "th": ["วันนี้ร้านไฟแรงมาก ⚡", "กำลังแพ็กของส่งเพียบเลย 🤝"],
        "id": ["Toko lagi semangat banget hari ini ⚡", "Pesanan lagi ramai dikirim 🤝"],
    }
    base = [t(lang, "smalltalk_hi")] + alt.get(lang, alt["vi"])
    follow = t(lang, "smalltalk_askback")
    raw = f"{random.choice(base)} {follow}"
    return rephrase_casual(raw, intent="generic", lang=lang, temperature=0.5)


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
        print(f"✅ {prefix} loaded: {len(meta)} chunks")
        return idx, meta
    except Exception as e:
        print(f"❌ Load index '{prefix}':", repr(e))
        return None, None


IDX_PROD, META_PROD = _safe_read_index("products")
IDX_POL,  META_POL  = _safe_read_index("policies")

# === Reload vectors (FAISS) ===
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
    t = (text or "").strip()
    if not t:
        return DEFAULT_LANG
    if re.search(r"[\u4e00-\u9fff]", t):  # CJK
        return "zh" if "zh" in SUPPORTED_LANGS else DEFAULT_LANG
    if re.search(r"[\u0E00-\u0E7F]", t):  # Thai
        return "th" if "th" in SUPPORTED_LANGS else DEFAULT_LANG
    # Vietnamese diacritics
    if re.search(r"[ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộơóờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵ]", t, flags=re.I):
        return "vi" if "vi" in SUPPORTED_LANGS else DEFAULT_LANG
    if re.search(r"\b(yang|dan|tidak|saja|terima|kasih)\b", t.lower()):
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
        "smalltalk_hi": "Mình vẫn ổn nè, đang trực chat hỗ trợ bạn đây 😊",
        "smalltalk_askback": "Bạn cần tìm món nào hôm nay để mình gợi ý nhanh nha?",
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
        "smalltalk_hi": "I’m doing great and ready to help 😊",
        "smalltalk_askback": "What are you looking for today so I can suggest fast?",
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
        "smalltalk_hi": "我很好，随时为你服务哦 😊",
        "smalltalk_askback": "今天想找什么？我帮你快速推荐～",
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
        "smalltalk_hi": "สบายดีมาก พร้อมช่วยเลยครับ/ค่ะ 😊",
        "smalltalk_askback": "วันนี้อยากหาสินค้าแบบไหน เดี๋ยวแนะนำให้เร็ว ๆ นะ",
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
        "smalltalk_hi": "Baik banget dan siap bantu 😊",
        "smalltalk_askback": "Hari ini cari apa? Biar saya rekomendasikan cepat ya.",
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


SYSTEM_STYLE = (
    "Bạn là trợ lý bán hàng Aloha tên là Aloha Bot. Tông giọng: thân thiện, chủ động, "
    "trả lời ngắn gọn như người thật; dùng 1–3 emoji hợp ngữ cảnh (không lạm dụng). "
    "Luôn dựa vào CONTEXT (nội dung RAG). Không bịa. Nếu thiếu dữ liệu thực tế, nói 'mình chưa có dữ liệu' "
    "và hỏi lại 1 câu để làm rõ. Trình bày dễ đọc: gạch đầu dòng khi liệt kê; 1 câu chốt hành động."
)

FEW_SHOT_EXAMPLES = [
    {"role":"user","content":[{"type":"input_text","text":"helo"}]},
    {"role":"assistant","content":[{"type":"input_text","text":"Xin chào 👋 Rất vui được phục vụ bạn! Bạn muốn mình giúp gì không nè? 🙂"}]},
    {"role":"user","content":[{"type":"input_text","text":"shop bạn có những gì"}]},
    {"role":"assistant","content":[{"type":"input_text","text":"Mời bạn tham quan cửa hàng tại đây ạ 🛍️ 👉 https://shop.aloha.id.vn/zh"}]},
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

    # sản phẩm & mô tả
    if any(k in t0 for k in PRODUCT_KEYWORDS): return "product"
    if "có bán" in t0 or "bán không" in t0 or "bán ko" in t0: return "product"
    if "có gì đặc biệt" in t0 or "điểm đặc biệt" in t0 or "có gì đặt biệt" in t0: return "product_info"

    return "other"

def build_messages(system, history, context, user_question):
    msgs = [{"role":"system","content":[{"type":"input_text","text":system}]}]
    msgs.extend(FEW_SHOT_EXAMPLES)
    for h in list(history)[-3:]:
        msgs.append({"role":h["role"], "content":[{"type":"input_text","text":h["content"]}]})
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
    "đồng hồ": ["dong ho","dong-ho","dongho","watch","watchface","galaxy watch","apple watch","amazfit","seiko","casio","nh35","nh36"],
    "dây đồng hồ": ["day dong ho","daydongho","watch band","band","strap","loop","bracelet"],
    "kính cường lực": ["kinh cuong luc","tempered glass","screen protector","full glass"],
    "ốp lưng": ["op lung","case","cover","bumper"],
    "áo thun": ["ao thun","ao phong","tshirt","t-shirt","tee"],
    "áo phông": ["ao phong","ao thun","tshirt","t-shirt","tee"],
    # Chinese / Thai / Indonesian
    "手表": ["腕表","watch","表带","钢化膜","保护壳"],
    "นาฬิกา": ["watch","สายนาฬิกา","ฟิล์มกระจก","เคส"],
    "jam tangan": ["watch","tali jam","pelindung layar","case"],
}

def _normalize_text(s: str) -> str:
    return re.sub(r"[^0-9a-z\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff\u0E00-\u0E7F ]", " ", (s or "").lower())

def _query_tokens(q: str, lang: str = "vi") -> set:
    qn = _normalize_text(q)
    words = [w for w in qn.split() if len(w) > 1]
    tokens = set(words)
    for i in range(len(words) - 1):
        tokens.add((words[i] + words[i+1]).strip())  # bigram

    joined = " ".join(words)
    combo_phrases = {
        "vi": ["đồng hồ","áo thun","áo phông","dây đồng hồ","kính cường lực","ốp lưng"],
        "en": ["watch band","screen protector","phone case","t shirt","t-shirt","watch"],
        "zh": ["手表","腕表","表带","钢化膜","手机壳"],
        "th": ["นาฬิกา","สายนาฬิกา","ฟิล์มกระจก","เคส"],
        "id": ["jam tangan","tali jam","pelindung layar","case"],
    }
    for phrase in combo_phrases.get(lang, []):
        if phrase in joined:
            tokens.add(_normalize_text(phrase).replace(" ", ""))
            for k, syns in VN_SYNONYMS.items():
                if _normalize_text(k) in _normalize_text(phrase):
                    for s in syns:
                        tokens.add(_normalize_text(s).replace(" ", ""))

    for synlist in VN_SYNONYMS.values():
        for s in synlist:
            if s in joined:
                tokens.add(_normalize_text(s).replace(" ", ""))
    return tokens

def filter_hits_by_query(hits, q, lang="vi"):
    """Giữ lại hit có ít nhất 1 token/cụm của câu hỏi xuất hiện trong title/tags/type/variant."""
    if not hits:
        return []
    qtoks = _query_tokens(q, lang=lang)
    kept = []
    for d in hits:
        hay = _normalize_text(" ".join([d.get("title",""), d.get("tags",""), d.get("product_type",""), d.get("variant","")]))
        hay_no_space = hay.replace(" ", "")
        ok = any((t in hay) or (t in hay_no_space) for t in qtoks)
        if ok:
            kept.append(d)
    return kept

def should_relax_filter(q: str, hits: list) -> bool:
    qn = _normalize_text(q)
    return len(qn.split()) <= 2 and len(hits) > 0

# ====== Compose trả lời ======
def compose_product_reply(hits, lang: str = "vi"):
    items = []
    for d in hits[:2]:
        title   = d.get("title") or "Sản phẩm"
        price   = d.get("price")
        variant = d.get("variant")
        stock   = _stock_line(d)
        line = f"• {title}"
        if variant: line += f" ({variant})"
        if price:   line += f" — {price} đ"
        line += f" — {stock}"
        items.append(line)
    if not items:
        return t(lang, "fallback")
    raw = f"{t(lang,'suggest_hdr')}\n" + "\n".join(items) + "\n\n" + t(lang,"product_pts")
    return rephrase_casual(raw, intent="product", lang=lang)

def compose_product_info(hits, lang: str = "vi"):
    if not hits:
        return t(lang, "fallback")
    d = hits[0]
    title = d.get("title") or "Sản phẩm"
    price = d.get("price")
    stock = _stock_line(d)
    bullets = _extract_features_from_text(d.get("text",""))
    body = "\n".join(bullets) if bullets else "• Thiết kế tối giản, dễ phối đồ\n• Chất liệu thoáng, dễ vệ sinh"
    price_line = f"Giá tham khảo: {price} đ" if price else ""
    raw = (
        f"{t(lang,'highlights', title=title)}\n"
        f"{body}\n"
        f"{price_line}\n"
        f"Tình trạng: {stock}\n"
        f"{t(lang,'product_pts')}"
    ).strip()
    return rephrase_casual(raw, intent="product", lang=lang)

def compose_contextual_answer(context, question, history):
    msgs = build_messages(SYSTEM_STYLE, history, context, question)
    _, reply = call_openai(msgs, temperature=0.6)
    return reply

def answer_with_rag(user_id, user_question):
    s = _get_sess(user_id)
    hist = s["hist"]

    intent = detect_intent(user_question)
    lang = detect_lang(user_question)
    print(f"🔎 intent={intent} | 🗣️ lang={lang}")

  # trong answer_with_rag, ngay sau print(...)
    if intent == "greet":
        return greet_text(lang), []

    if intent == "smalltalk":               # <-- đưa smalltalk lên trước
        return handle_smalltalk(user_question, lang=lang), []

    if intent == "browse":
        url = SHOP_URL_MAP.get(lang, SHOP_URL_MAP.get(DEFAULT_LANG, SHOP_URL))
        return t(lang, "browse", url=url), []

    if intent == "new_items":
        items = get_new_arrivals(days=30, topk=4)
        return compose_new_arrivals(lang=lang, items=items), items[:2]


    prod_hits, prod_scores = search_products_with_scores(user_question, topk=8)
    best = max(prod_scores or [0.0])

    filtered_hits = filter_hits_by_query(prod_hits, user_question, lang=lang) if STRICT_MATCH else prod_hits
    if STRICT_MATCH and not filtered_hits and should_relax_filter(user_question, prod_hits):
        print("🔧 relaxed_filter=True (fallback to unfiltered hits)")
        filtered_hits = prod_hits

    print(f"📈 best_score={best:.3f}, hits={len(prod_hits)}, kept_after_filter={len(filtered_hits)}")
    context = retrieve_context(user_question, topk=6)

    if intent == "policy" and context:
        ans = compose_contextual_answer(context, user_question, hist)
        ans = f"{t(lang,'policy_hint')} {ans}"
        return rephrase_casual(ans, intent="policy", temperature=0.5, lang=lang), []

    if intent in {"product","product_info","other"}:
        if not filtered_hits or best < SCORE_MIN:
            url = SHOP_URL_MAP.get(lang, SHOP_URL_MAP.get(DEFAULT_LANG, SHOP_URL))
            return (t(lang, "oos", url=url)), []

    if intent == "product_info":
        return compose_product_info(filtered_hits, lang=lang), filtered_hits[:1]

    if intent in {"product","other"} and filtered_hits and best >= SCORE_MIN:
        return compose_product_reply(filtered_hits, lang=lang), filtered_hits[:2]

    if context:
        ans = compose_contextual_answer(context, user_question, hist)
        return rephrase_casual(ans, intent="generic", temperature=0.7, lang=lang), []
    
    

    return (t(lang, "fallback")), []

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
