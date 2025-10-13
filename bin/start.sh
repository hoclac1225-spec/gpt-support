#!/usr/bin/env bash
set -e

# Mặc định thư mục
export DATA_DIR=${DATA_DIR:-./data}
export VECTOR_DIR=${VECTOR_DIR:-./vectors}
mkdir -p "$VECTOR_DIR"

echo "==> Preflight: build vectors if missing..."

need_rebuild=false
[ ! -f "$VECTOR_DIR/products.index" ] && need_rebuild=true
[ ! -f "$VECTOR_DIR/policies.index" ] && need_rebuild=true

# Build policies nếu thiếu
if [ ! -f "$VECTOR_DIR/policies.index" ]; then
  if [ -f "ingest_policy.py" ]; then
    echo " -> Building policy index..."
    python ingest_policy.py --file "$DATA_DIR/policies.md" || true
  fi
fi

# Build products nếu thiếu và có SHOPIFY creds
if [ ! -f "$VECTOR_DIR/products.index" ]; then
  if [ -n "$SHOPIFY_STORE" ] && [ -n "$SHOPIFY_ADMIN_API_TOKEN" ]; then
    echo " -> Building product index from Shopify..."
    python ingest_products.py || true
  else
    echo " -> Skip product ingest (missing SHOPIFY creds)."
  fi
fi

echo "==> Starting web app..."
# Bạn có thể dùng gunicorn để ổn định hơn:
# exec gunicorn app:app --preload --workers=2 --threads=4 --timeout=120 --bind 0.0.0.0:${PORT:-3000}
exec python app.py
