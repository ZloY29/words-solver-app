import os
from flask import Flask, send_from_directory
from dotenv import load_dotenv
from werkzeug.exceptions import NotFound

from backend.words_solver.dictionary import load_words, build_trie
from backend.words_solver.model import load_letter_model
from backend.words_solver.routes import create_routes


def create_app() -> Flask:
    load_dotenv()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    model_path = os.path.join(assets_dir, "letter_recognition_model.h5")
    dictionary_path = os.path.join(assets_dir, "cleaned_filtered_russian_words.json")

    app = Flask(__name__, static_folder="static")

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
