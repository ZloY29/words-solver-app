import logging
import os
import warnings

from dotenv import load_dotenv
from flask import Flask, request, send_from_directory
from werkzeug.exceptions import NotFound

from backend.words_solver.dictionary import build_trie, load_words
from backend.words_solver.logging_config import configure_logging
from backend.words_solver.model import load_letter_model
from backend.words_solver.routes import create_routes

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    load_dotenv()
    configure_logging()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=FutureWarning)

    model_path = os.path.join(assets_dir, "letter_recognition_model.h5")
    dictionary_path = os.path.join(assets_dir, "cleaned_filtered_russian_words.json")

    app = Flask(__name__, static_folder="static")

    @app.before_request
    def _before_request():
        logger.info(
            "request start method=%s path=%s",
            getattr(request, "method", "?"),
            getattr(request, "path", "?"),
        )

    @app.after_request
    def _after_request(response):
        logger.info("request end status=%s", response.status_code)
        return response

    model = load_letter_model(model_path)
    words = load_words(dictionary_path)
    trie = build_trie(words)

    app.register_blueprint(create_routes(model=model, trie=trie))

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        try:
            return send_from_directory(app.static_folder, path)
        except NotFound:
            return send_from_directory(app.static_folder, "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
