# tokenizer.py
import json
import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.merges = {}  # optional, falls du echtes BPE nutzt

    # ---------------------------------------------------------
    # TRAIN (falls du schon eine train-Methode hast -> behalten)
    # ---------------------------------------------------------
    def train(self, text: str):
        # Simple char-level fallback training:
        # Wenn dein alter Code echtes BPE hat, nutze den!
        chars = sorted(list(set(text)))
        self.vocab = chars[: self.vocab_size]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    # ---------------------------------------------------------
    # ENCODE / DECODE (falls vorhanden -> behalten)
    # ---------------------------------------------------------
    def encode(self, text: str):
        # Simple char fallback
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    # ---------------------------------------------------------
    # ✅ NEU: SAVE / LOAD  (DAS ist der Fix)
    # ---------------------------------------------------------
    def save(self, path="tokenizer.json"):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": self.itos,
            "merges": self.merges
        }
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path="tokenizer.json"):
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data.get("vocab_size", 4096))
        tok.vocab = data.get("vocab", [])
        tok.stoi = data.get("stoi", {})
        tok.itos = {int(k): v for k, v in data.get("itos", {}).items()} if isinstance(list(data.get("itos", {}).keys())[0], str) else data.get("itos", {})
        tok.merges = data.get("merges", {})
        return tok
