import json

from .trie import Trie


def load_words(dictionary_path):
    with open(dictionary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_trie(words):
    trie = Trie()
    for w in words:
        trie.insert(w)
    return trie
