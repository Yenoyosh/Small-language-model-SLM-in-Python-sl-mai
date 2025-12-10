# train.py
import os
import time
import torch
from torch.utils.data import DataLoader
from tokenizer import BPETokenizer
from data import TextDataset
from model import MiniGPT

# ---------------------------
# EINSTELLUNGEN (Speed!)
# ---------------------------

def ask_int(prompt, default):
    """Hilfsfunktion: fragt Zahl ab, nutzt default bei leer/Fehler."""
    s = input(f"{prompt} (default={default}): ")
    s = s.strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        print(f"Ungültige Eingabe, nehme default={default}.")
        return default

TOTAL_EPOCHS = ask_int("Gib die Gesamtanzahl der Epochen an", 40)
BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 64
LR = 3e-4

MAX_BATCHES_PER_EPOCH = ask_int("Gib die Batches pro Epoche an", 250)

CHECKPOINT_FILE = "checkpoint.pt"
MODEL_FILE = "minigpt_grundwissen.pt"
TOKENIZER_FILE = "tokenizer.json"

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Verwendetes Device:", device, flush=True)

# ---------------------------
# TEXT LADEN
# ---------------------------
with open("grundwissen.txt", "r", encoding="utf8") as f:
    text = f.read()

print("Textlänge in Zeichen:", len(text), flush=True)

# ---------------------------
# ✅ TOKENIZER NUR EINMAL TRAINIEREN
# ---------------------------
if os.path.exists(TOKENIZER_FILE):
    tok = BPETokenizer.load(TOKENIZER_FILE)
    print("Tokenizer geladen.", flush=True)
else:
    tok = BPETokenizer(vocab_size=4096)
    tok.train(text)
    tok.save(TOKENIZER_FILE)
    print("Tokenizer neu trainiert und gespeichert.", flush=True)

encoded = tok.encode(text)
print("Anzahl Tokens:", len(encoded), flush=True)

# ---------------------------
# DATASET
# ---------------------------
block_size = DEFAULT_BLOCK_SIZE
dataset = TextDataset(encoded, block_size=block_size)

if len(dataset) <= 0:
    print("WARNUNG: Dataset zu klein – block_size wird reduziert.", flush=True)
    block_size = max(8, len(encoded) // 4)
    dataset = TextDataset(encoded, block_size=block_size)

print("Dataset-Länge:", len(dataset), flush=True)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Windows-safe
)

# ---------------------------
# MODELL + OPTIMIZER
# ---------------------------
model = MiniGPT(vocab_size=len(tok.vocab), max_len=block_size)
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------
# CHECKPOINT LADEN (RESUME)
# ---------------------------
start_epoch = 0
if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Checkpoint geladen – Weitertraining ab Epoche {start_epoch}", flush=True)
else:
    print("Kein Checkpoint gefunden – starte neues Training.", flush=True)

# ---------------------------
# TRAINING (MINI-EPOCHEN)
# ---------------------------
model.train()
print("Training startet jetzt (Mini-Epochen)...", flush=True)

global_start = time.time()

for epoch in range(start_epoch, TOTAL_EPOCHS):
    epoch_start = time.time()
    total_loss = 0.0
    last_print = time.time()

    for i, (x, y) in enumerate(loader):
        if i >= MAX_BATCHES_PER_EPOCH:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

        # jede Sekunde Status
        now = time.time()
        if now - last_print >= 1.0:
            batches_done = i + 1
            avg_loss_so_far = total_loss / batches_done

            elapsed_epoch = now - epoch_start
            speed = batches_done / max(elapsed_epoch, 1e-6)
            remaining = MAX_BATCHES_PER_EPOCH - batches_done
            eta_sec = remaining / max(speed, 1e-6)

            print(
                f"Epoch {epoch}/{TOTAL_EPOCHS-1} | "
                f"Batch {batches_done}/{MAX_BATCHES_PER_EPOCH} | "
                f"loss={loss.item():.4f} | avg_loss={avg_loss_so_far:.4f} | "
                f"ETA ~{eta_sec:.0f}s",
                flush=True
            )
            last_print = now

    avg_loss = total_loss / max(1, (i + 1))
    epoch_time = time.time() - epoch_start

    print(
        f"✅ Epoch {epoch} fertig (Mini) | avg_loss={avg_loss:.4f} | "
        f"Zeit: {epoch_time:.1f}s",
        flush=True
    )

    # Autosave nach jeder Mini-Epoche
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "block_size": block_size,
        "vocab_size": len(tok.vocab)
    }, CHECKPOINT_FILE)

    torch.save(model.state_dict(), MODEL_FILE)
    print("💾 Checkpoint gespeichert.\n", flush=True)

total_time = time.time() - global_start
print(f"🏁 Training komplett! Gesamtzeit: {total_time/60:.1f} min", flush=True)
print("Modell gespeichert als", MODEL_FILE, flush=True)
