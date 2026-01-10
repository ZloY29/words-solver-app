import json

from flask import Response

from .logging_config import request_id_var


def json_response(payload, status=200):
    body = json.dumps(payload, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")


def error_response(code, message, status=400):
    rid = request_id_var.get() or "-"
    payload = {
        "error": message,
        "error_code": code,
        "request_id": rid,
    }
    return json_response(payload, status=status)
