@echo off
setlocal
set "API_BASE=https://gpt-support-1.onrender.com"
set "ADMIN_TOKEN=YOUR_ADMIN_TOKEN_HERE"

echo === Health ===
curl -s "%API_BASE%/health" & echo.

echo === Rebuild vectors ===
curl -s -X POST "%API_BASE%/admin/rebuild_vectors_now?token=%ADMIN_TOKEN%" & echo.

echo === RAG status ===
curl -s "%API_BASE%/debug/rag_status" & echo.

echo === Quick search ===
curl -s "%API_BASE%/api/product_search?q=day%%20dong%%20ho&debug=1" & echo.
endlocal
