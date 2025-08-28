# -*- coding: utf-8 -*-
import os, requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)

# Lấy danh sách domain cho phép từ .env
allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins:
    origins = [o.strip() for o in allowed_origins.split(",")]
else:
    origins = ["*"]  # fallback (không nên dùng trong production)

# Bật CORS chỉ cho domain trong ALLOWED_ORIGINS
CORS(app, origins=origins)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])

    payload = {
        "model": "gpt-4o-mini",
        "input": [
            {"role": "system", "content": [
                {"type": "input_text", "text": "You are a helpful support agent. Answer clearly and briefly."}
            ]},
            *messages
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    return jsonify(r.json())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(host="127.0.0.1", port=port, debug=True)
