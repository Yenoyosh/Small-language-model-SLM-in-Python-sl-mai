# gui_generate.py

print("KI wird geladen...")

import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk

from model import MiniGPT
from tokenizer import BPETokenizer
from memory import (
    load_memory,
    save_memory,
    add_prompt,
    build_style_profile,
    style_similarity,
)

MODEL_FILE = "minigpt_grundwissen.pt"
CHECKPOINT_FILE = "checkpoint.pt"
TOKENIZER_FILE = "tokenizer.json"

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Verwendetes Device:", device)

# ---------------------------
# Tokenizer laden
# ---------------------------
tok = BPETokenizer.load(TOKENIZER_FILE)

# ---------------------------
# block_size aus Checkpoint holen
# ---------------------------
block_size = 64  # Fallback
try:
    ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
    block_size = int(ckpt.get("block_size", block_size))
except Exception as e:
    print("Warnung: Konnte block_size nicht aus checkpoint lesen:", e)

print("Verwendete block_size/max_len:", block_size)

# ---------------------------
# Modell laden
# ---------------------------
model = MiniGPT(vocab_size=len(tok.vocab), max_len=block_size)
state_dict = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---------------------------
# Memory + Stilprofil
# ---------------------------
memory = load_memory()
style_profile = build_style_profile(memory)
print(f"Anzahl gespeicherter Prompts: {len(memory)}")


# ---------------------------
# Sampling
# ---------------------------
def sample_next_id(logits, temperature=0.4, top_k=30):
    logits = logits / max(temperature, 1e-6)

    if top_k is not None and 0 < top_k < logits.size(-1):
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        probs = F.softmax(values, dim=-1)
        next_local = torch.multinomial(probs, num_samples=1)
        next_id = indices.gather(-1, next_local)
    else:
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

    return next_id  # (1, 1)


def generate_one(prompt, steps=80, temperature=0.4):
    tokens = tok.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)

    if idx.size(1) > block_size:
        idx = idx[:, -block_size:]

    with torch.no_grad():
        for _ in range(steps):
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
            logits = model(idx)[:, -1, :]
            next_id = sample_next_id(logits, temperature=temperature, top_k=30)
            idx = torch.cat([idx, next_id], dim=1)

    return tok.decode(idx[0].tolist())


# ---------------------------
# Scoring: Grammatik + Stilprofil
# ---------------------------
GERMAN_COMMON_WORDS = [
    "und", "oder", "aber", "der", "die", "das", "ein", "eine",
    "ist", "sind", "war", "waren", "haben", "hat", "nicht",
    "mathematik", "wissenschaft", "natur", "computer", "energie",
]


def score_candidate(prompt, text, style_weight, style_profile_local):
    score = 0.0
    t = text.strip()

    # 1. Großbuchstabe am Anfang
    if t and t[0].isupper():
        score += 1.0

    # 2. Satzzeichen am Ende
    if t.endswith((".", "!", "?")):
        score += 1.0

    # 3. deutsche Wörter
    lower = t.lower()
    for w in GERMAN_COMMON_WORDS:
        if w in lower:
            score += 0.3

    # 4. Wiederholungen bestrafen
    max_run = 1
    current_run = 1
    for i in range(1, len(t)):
        if t[i] == t[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run >= 5:
        score -= (max_run - 4) * 0.5

    # 5. Stil-Ähnlichkeit (POS / Satzstruktur)
    if style_profile_local and style_weight > 0:
        sim = style_similarity(t, style_profile_local)  # 0..1
        score += style_weight * 3.0 * sim

    return score


# ---------------------------
# Tkinter-GUI
# ---------------------------
root = tk.Tk()
root.title("Mini-GPT Deutsch (Prompt-Stil & Satzstruktur)")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Prompt
ttk.Label(frame, text="Dein Prompt:").grid(row=0, column=0, sticky="w")
prompt_text = tk.Text(frame, height=4, width=60)
prompt_text.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=5)

# Stil-Einfluss
ttk.Label(frame, text="Stil-Einfluss (0 = aus, 1 = stark):").grid(row=2, column=0, sticky="w", pady=(10, 0))
style_var = tk.DoubleVar(value=0.5)
style_slider = ttk.Scale(frame, from_=0.0, to=1.0, orient="horizontal", variable=style_var)
style_slider.grid(row=2, column=1, sticky="we", padx=5)

# Anzahl interner Kandidaten
ttk.Label(frame, text="interne Kandidaten (1–10):").grid(row=3, column=0, sticky="w", pady=(10, 0))
cand_var = tk.IntVar(value=5)
cand_spin = ttk.Spinbox(frame, from_=1, to=10, textvariable=cand_var, width=5)
cand_spin.grid(row=3, column=1, sticky="w", padx=5)

# Temperatur
ttk.Label(frame, text="Temperatur (0.3–0.7, Punkt statt Komma):").grid(row=4, column=0, sticky="w", pady=(10, 0))
temp_var = tk.DoubleVar(value=0.4)
temp_entry = ttk.Entry(frame, textvariable=temp_var, width=6)
temp_entry.grid(row=4, column=1, sticky="w", padx=5)

# Ausgabe
ttk.Label(frame, text="Ausgabe:").grid(row=6, column=0, sticky="w")
output_text = tk.Text(frame, height=18, width=80)
output_text.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=5)

for r in range(8):
    frame.rowconfigure(r, weight=0)
frame.rowconfigure(7, weight=1)
for c in range(4):
    frame.columnconfigure(c, weight=1)


# ---------------------------
# Generate-Button-Logik
# ---------------------------
def on_generate():
    global memory, style_profile

    prompt = prompt_text.get("1.0", "end").strip()
    if not prompt:
        output_text.delete("1.0", "end")
        output_text.insert("1.0", "Bitte einen Prompt eingeben.")
        return

    try:
        style_w = float(style_var.get())
    except ValueError:
        style_w = 0.5

    try:
        temperature = max(0.1, float(temp_var.get()))
    except ValueError:
        temperature = 0.4

    try:
        n_cand = max(1, min(10, int(cand_var.get())))
    except ValueError:
        n_cand = 3

    scored = []
    for _ in range(n_cand):
        text = generate_one(prompt, steps=80, temperature=temperature)
        s = score_candidate(prompt, text, style_w, style_profile)
        scored.append((s, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_text = scored[0]

    output_text.delete("1.0", "end")
    output_text.insert(
        "1.0",
        f"Beste Antwort (Score {best_score:.2f}):\n\n{best_text}\n",
    )

    # Prompt in Memory aufnehmen und Stilprofil aktualisieren
    memory = add_prompt(memory, prompt)
    save_memory(memory)
    style_profile = build_style_profile(memory)


generate_button = ttk.Button(frame, text="Generieren", command=on_generate)
generate_button.grid(row=5, column=0, pady=10, sticky="w")


def on_close():
    save_memory(memory)
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
