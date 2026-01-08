import json
from flask import Response


def json_response(payload, status=200):
    body = json.dumps(payload, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")
