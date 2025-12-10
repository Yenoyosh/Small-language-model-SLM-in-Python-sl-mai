# memory.py
import json
import os
import math
from collections import defaultdict

MEMORY_FILE = "memory.json"

# sehr kleine Listen für deutsche Funktionswörter
PREPOSITIONS = {
    "in", "an", "auf", "unter", "über", "vor", "hinter", "neben",
    "zwischen", "mit", "ohne", "für", "von", "zu", "nach", "gegen",
    "bei", "seit", "bis", "durch", "um", "entlang", "aus"
}

CONJUNCTIONS = {
    "und", "oder", "aber", "denn", "doch", "weil", "da", "obwohl",
    "während", "dass", "damit", "sodass", "wenn", "falls"
}

PRONOUNS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "sie",
    "mich", "dich", "ihn", "uns", "euch",
    "mein", "dein", "sein", "ihr", "unser", "euer"
}

# einige typische Adverbien
ADVERBS = {
    "oft", "selten", "manchmal", "immer", "nie",
    "heute", "gestern", "morgen", "bald", "früher",
    "sehr", "kaum", "genau", "ungefähr"
}


def _tokenize(text: str):
    tokens = []
    for raw in text.replace("\n", " ").split():
        tok = raw.strip(".,;:!?\"'()[]{}«»„“")
        if tok:
            tokens.append(tok)
    return tokens


def extract_style_features(text: str):
    """
    Extrahiert einfache Stil-Features:
    - durchschnittliche Satzlänge
    - durchschnittliche Wortlänge
    - Anteil (ungefähr) Nomen / Adjektive / Adverbien / Präpositionen
    - Anteil Funktionswörter
    - Anteil Satzzeichen
    """
    tokens = _tokenize(text)
    n_tokens = len(tokens)
    n_chars = len(text)

    if n_tokens == 0 or n_chars == 0:
        return {
            "avg_sent_len": 0.0,
            "avg_token_len": 0.0,
            "noun_ratio": 0.0,
            "adj_ratio": 0.0,
            "adv_ratio": 0.0,
            "prep_ratio": 0.0,
            "func_ratio": 0.0,
            "punct_ratio": 0.0,
        }

    # Sätze zählen über ., !, ?
    n_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    avg_sent_len = n_tokens / n_sentences
    avg_token_len = sum(len(t) for t in tokens) / n_tokens

    noun_like = 0
    adj_like = 0
    adv_like = 0
    prep = 0
    func = 0

    for i, tok in enumerate(tokens):
        lower = tok.lower()

        # Präpositionen / Funktionswörter
        if lower in PREPOSITIONS:
            prep += 1
            func += 1
        if lower in CONJUNCTIONS or lower in PRONOUNS:
            func += 1

        # sehr grobe Heuristiken:
        # Nomen: großgeschrieben, nicht erstes Wort
        if i > 0 and tok[0].isupper():
            noun_like += 1

        # Adjektive (Endungen)
        if lower.endswith(("ig", "lich", "isch", "los", "voll", "sam")):
            adj_like += 1

        # Adverbien (Endungen + Liste)
        if lower in ADVERBS or lower.endswith(("erweise", "erweise", "lich")):
            adv_like += 1

    punct_ratio = sum(1 for ch in text if ch in ".,;:!?") / n_chars

    return {
        "avg_sent_len": avg_sent_len,
        "avg_token_len": avg_token_len,
        "noun_ratio": noun_like / n_tokens,
        "adj_ratio": adj_like / n_tokens,
        "adv_ratio": adv_like / n_tokens,
        "prep_ratio": prep / n_tokens,
        "func_ratio": func / n_tokens,
        "punct_ratio": punct_ratio,
    }


def load_memory():
    """
    Lädt die gespeicherten Prompts.
    Alte Versionen mit [{"prompt":..., "answer":...}] werden in eine
    einfache Prompt-Liste umgewandelt.
    """
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf8") as f:
                data = json.load(f)

            prompts = []
            if isinstance(data, list):
                if data and isinstance(data[0], dict) and "prompt" in data[0]:
                    # alte Struktur -> nur Prompts übernehmen
                    for item in data:
                        p = item.get("prompt", "")
                        if p.strip():
                            prompts.append(p.strip())
                elif data and isinstance(data[0], str):
                    prompts = [s.strip() for s in data if isinstance(s, str) and s.strip()]
            else:
                prompts = []

            return prompts
        except Exception:
            pass
    return []  # Fallback


def save_memory(memory):
    """Speichert nur die Liste deiner Prompts."""
    with open(MEMORY_FILE, "w", encoding="utf8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_prompt(memory, prompt, max_len=2000):
    """Fügt einen neuen Prompt hinzu."""
    prompt = (prompt or "").strip()
    if not prompt:
        return memory
    memory.append(prompt)
    if len(memory) > max_len:
        memory = memory[-max_len:]
    return memory


def build_style_profile(memory):
    """
    Baut ein gemitteltes Stilprofil aus allen bisherigen Prompts.
    Gibt ein Dict {feature_name -> durchschnittlicher Wert} zurück.
    """
    if not memory:
        return None

    acc = defaultdict(float)
    count = 0

    for prompt in memory:
        feats = extract_style_features(prompt)
        for k, v in feats.items():
            acc[k] += v
        count += 1

    if count == 0:
        return None

    return {k: v / count for k, v in acc.items()}


def style_similarity(text, style_profile):
    """
    Cosine-Similarity zwischen den Stil-Features deines Textes
    und dem gemittelten Stilprofil aus deinen Prompts.
    """
    if not style_profile:
        return 0.0

    feats = extract_style_features(text)
    keys = list(style_profile.keys())

    num = sum(feats[k] * style_profile[k] for k in keys)
    norm1 = math.sqrt(sum(feats[k] ** 2 for k in keys))
    norm2 = math.sqrt(sum(style_profile[k] ** 2 for k in keys))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return num / (norm1 * norm2)
