import json as _json
from flask import Blueprint, request, jsonify, Response, stream_with_context
from agent import ask_stream

bp = Blueprint("scholar", __name__)

@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@bp.route("/stream", methods=["POST"])
def stream_endpoint():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    question   = data.get("question", "").strip()
    thread_id  = str(data.get("thread_id", "1")).strip()
    web_search = bool(data.get("web_search", False))

    if not question:
        return jsonify({"error": "Missing required field: question"}), 400

    _STATUS_PREFIX = "\x00STATUS\x00"

    def generate():
        try:
            for token in ask_stream(query=question, thread_id=thread_id, web_search=web_search):
                if token.startswith(_STATUS_PREFIX):
                    msg = token[len(_STATUS_PREFIX):]
                    yield f"data: {_json.dumps({'status': msg})}\n\n"
                else:
                    yield f"data: {_json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )