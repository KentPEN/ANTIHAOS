
import os
import base64
import re
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
VISION_MODEL = "qwen/qwen3.8-27b"

MAX_IMAGE_SIZE = 5 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 9 * 1024 * 1024

client = Groq(api_key=API_KEY) if API_KEY else None


def validate_image(data_url):
    if not isinstance(data_url, str):
        raise ValueError("Некорректный формат изображения.")

    match = re.fullmatch(
        r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)",
        data_url
    )

    if not match:
        raise ValueError("Поддерживаются только JPG, PNG и WEBP.")

    mime = match.group(1)
    encoded = match.group(2)

    if mime not in ALLOWED_MIME:
        raise ValueError("Этот формат изображения не поддерживается.")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception:
        raise ValueError("Не удалось прочитать изображение.")

    if not image_bytes or len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Размер фотографии не должен превышать 5 МБ.")

    return data_url


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "provider": "Groq",
        "model": MODEL,
        "vision_model": VISION_MODEL,
        "ai_configured": bool(API_KEY)
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({
            "error": "Не найден GROQ_API_KEY. Проверь файл .env."
        }), 503

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    language = data.get("language", "ru")

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Сообщение не найдено."}), 400

    if language not in ("ru", "ko"):
        language = "ru"

    system_prompt = """
Ты — ANTIХАОС, интеллектуальный помощник.
Помогай людям разбираться в сложных ситуациях,
находить ясность и превращать проблемы в конкретные шаги.

Правила:
1. Отвечай на языке пользователя.
2. Если выбран корейский язык — отвечай по-корейски.
3. Будь внимательным, спокойным и доброжелательным.
4. Не осуждай пользователя.
5. Не выдумывай факты.
6. Не перегружай ответы лишней информацией.
7. Предлагай конкретные и выполнимые действия.
8. Если пользователь прислал изображение, внимательно изучи его
   и отвечай с учётом того, что действительно видно.
9. Если изображение неясное, прямо скажи об этом.
10. Не называй себя Gemini, ChatGPT или Groq.
"""

    prepared = []

    for message in messages[-12:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")

        if role not in ("user", "assistant", "model"):
            continue

        if not isinstance(content, str):
            continue

        if not content.strip() and not message.get("image"):
            continue

        prepared.append({
            "role": "user" if role == "user" else "assistant",
            "content": content[:5000],
            "image": message.get("image") if role == "user" else None
        })

    if not prepared:
        return jsonify({"error": "Не удалось прочитать сообщение."}), 400

    # Используем последнее изображение из истории диалога.
    last_image_index = None

    for index, message in enumerate(prepared):
        if message["image"]:
            last_image_index = index

    groq_messages = [{
        "role": "system",
        "content": system_prompt
    }]

    try:
        for index, message in enumerate(prepared):
            content = message["content"] or (
                "Проанализируй прикреплённое изображение."
                if language == "ru"
                else "첨부된 이미지를 분석해 주세요."
            )

            if index == last_image_index:
                image_data = validate_image(message["image"])

                content = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data}
                    }
                ]

            groq_messages.append({
                "role": message["role"],
                "content": content
            })

        selected_model = VISION_MODEL if last_image_index is not None else MODEL

        result = client.chat.completions.create(
            model=selected_model,
            messages=groq_messages,
            max_completion_tokens=1200
        )

        reply = result.choices[0].message.content

        if not reply:
            return jsonify({
                "error": "Модель вернула пустой ответ. Попробуй ещё раз."
            }), 502

        return jsonify({
            "reply": reply,
            "provider": "Groq",
            "model": selected_model
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        print("GROQ ERROR:", type(error).__name__, str(error))

        return jsonify({
            "error": "Ошибка Groq API: " + str(error)[:500]
        }), 502


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "Запрос слишком большой. Уменьши размер фотографии."
    }), 413



@app.get("/yandex_4ded8897e99c4224.html")
def yandex_verification():
    return send_from_directory(BASE_DIR, "yandex_4ded8897e99c4224.html", mimetype="text/html")

if __name__ == "__main__":
    print("=" * 45)
    print("ANTIХАОС AI SERVER")
    print("Provider: Groq")
    print("Text model:", MODEL)
    print("Vision model:", VISION_MODEL)
    print("API configured:", bool(API_KEY))
    print("Address: http://127.0.0.1:5000")
    print("=" * 45)

    app.run(host="127.0.0.1", port=5000, debug=False)
