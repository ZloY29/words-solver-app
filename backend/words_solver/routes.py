import os
import tempfile

from flask import Blueprint, request
from PIL import Image

from .http import json_response
from .image_processing import process_image


def create_routes(model, trie):
    bp = Blueprint("routes", __name__)

    @bp.post("/upload")
    def upload_file():
        if "image" not in request.files:
            return json_response({"error": "No image part"}, status=400)

        file = request.files["image"]
        if not file.filename:
            return json_response({"error": "No selected file"}, status=400)

        tmp_path = None
        try:
            img = Image.open(file.stream)
            img = img.convert("RGB")
            img = img.resize((590, 1280), Image.Resampling.LANCZOS)

            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            img.save(tmp_path, format="JPEG", quality=95)

            result = process_image(tmp_path, model=model, trie=trie)
            return json_response(result, status=200)

        except Exception as e:
            return json_response({"error": str(e)}, status=500)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return bp
