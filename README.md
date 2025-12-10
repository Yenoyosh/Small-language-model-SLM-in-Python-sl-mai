# Small language model (SLM) in Python – sl mai

Ein kleines deutsches Sprachmodell (SLM), gebaut in Python und PyTorch.  
Das Modell kann lokal trainiert werden, verwendet einen eigenen BPE-Tokenizer und bietet eine Tkinter-GUI zur Textgenerierung – mit optionalem Kontext & Stilprofil.

---

## 📁 Enthaltene Dateien

Dieses Repository enthält **nur den Code**, nicht die Trainingsdaten und nicht das fertige Modell.

**Python-Dateien:**

- `train.py` – Training des Modells auf `grundwissen.txt`, erzeugt:
  - `tokenizer.json`
  - `checkpoint.pt`
  - `minigpt_grundwissen.pt`
- `model.py` – MiniGPT-Modellarchitektur (kleiner Transformer)
- `tokenizer.py` – einfacher BPE-Tokenizer
- `data.py` – Dataset, das Trainingssamples aus dem Text erzeugt
- `memory.py` – speichert Prompts & Stilprofil (`memory.json`)
- `context_manager.py` – optionaler Kontextmanager für die GUI
- `ai-V1-without-context.py` – einfache GUI ohne Kontext
- `sl-mai-ai-V2-with-context.py` – erweiterte GUI mit Stil, Reranking und Kontextoption

**Nicht im Repository enthalten (wird lokal erstellt oder muss erstellt werden):**

- `grundwissen.txt`
- `checkpoint.pt`
- `minigpt_grundwissen.pt`
- `tokenizer.json`
- `memory.json`

---

## 📄 Voraussetzungen

- Python **3.10+** (getestet mit Python 3.13)
- PyTorch (CPU-Version ausreichend)
- Tkinter (bei Windows-Python meist vorinstalliert)

### Installation der benötigten Module:

```bash
pip install torch
