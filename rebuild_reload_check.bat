@echo off
setlocal
set BASE=https://gpt-support-1.onrender.com
set TOKEN=aloha_admin_123
set ORIGIN=https://shop.aloha.id.vn

echo [1/4] Rebuild vectors...
curl -sS -X POST "%BASE%/admin/rebuild_vectors_now?token=%TOKEN%"
echo.

echo [2/4] Reload in-memory index...
curl -sS -X POST "%BASE%/admin/reload_vectors?token=%TOKEN%"
echo.

echo [3/4] RAG status:
curl -sS "%BASE%/debug/rag_status"
echo.

echo [4/4] Sample product_search (CORS check):
curl -sS -i "%BASE%/api/product_search?q=crepe&debug=1" -H "Origin: %ORIGIN%"
echo.
pause
endlocal
