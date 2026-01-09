import os
import tempfile

from flask import Blueprint, request
from PIL import Image

from .http import json_response
from .image_processing import process_image

import logging
import time
logger = logging.getLogger(__name__)


def create_routes(model, trie):
    bp = Blueprint("routes", __name__)

    @bp.get("/api/documents")
    def documents():
        return json_response({"documents": []}, status=200)

    @bp.post("/upload")
    def upload_file():
        start = time.perf_counter()

        if "image" not in request.files:
            logger.info("upload missing image part")
            return json_response({"error": "No image part"}, status=400)

        file = request.files["image"]
        if not file.filename:
            logger.info("upload empty filename")
            return json_response({"error": "No selected file"}, status=400)

        logger.info(
            "upload start filename=%s content_type=%s",
            file.filename,
            getattr(file, "content_type", None),
        )

        tmp_path = None
        try:
            img = Image.open(file.stream)
            img = img.convert("RGB")
            img = img.resize((590, 1280), Image.Resampling.LANCZOS)

            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            img.save(tmp_path, format="JPEG", quality=95)

            result = process_image(tmp_path, model=model, trie=trie)

            logger.info("upload ok seconds=%.3f", time.perf_counter() - start)
            return json_response(result, status=200)

        except Exception as e:
            # Full stacktrace into logs
            logger.exception("upload failed seconds=%.3f", time.perf_counter() - start)
            # Keep current behavior (do not break frontend)
            return json_response({"error": str(e)}, status=500)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.exception("failed to remove tmp file tmp_path=%s", tmp_path)

    return bp
