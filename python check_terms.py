# check_terms.py
import json, re, sys, os

PATH = r'vectors\products.meta.json'  # đổi nếu bạn để chỗ khác

TERMS = [
    '千層','千层','榴槤','榴莲','可麗餅','可丽饼','奶茶',
    'เครป','ทุเรียน','ชานม',
    'teh susu','boba','durian','crepe'
]

def has_term(d, t):
    blob = " ".join([
        d.get('title',''),
        d.get('tags',''),
        d.get('product_type',''),
        d.get('variant',''),
        d.get('text','')
    ]).lower()
    return t.lower() in blob

if not os.path.exists(PATH):
    print(f"⚠ File không tồn tại: {PATH}")
    sys.exit(1)

with open(PATH, encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} chunks from {PATH}\n")

for t in TERMS:
    hits = [d for d in data if has_term(d, t)]
    print(f"{t}: {len(hits)}")

print("\n--- samples (tối đa 5 mỗi term) ---")
for t in TERMS:
    hits = [(d.get('title',''), d.get('url','')) for d in data if has_term(d, t)]
    if hits:
        print(f"\n[{t}]")
        for title, url in hits[:5]:
            print("-", (title or "—")[:100], "|", url or "—")
