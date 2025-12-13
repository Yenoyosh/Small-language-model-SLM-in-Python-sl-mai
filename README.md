# 🧠 SL-MAI – Ein kleines deutsches Sprachmodell (SLM) mit Python

SL-MAI ist ein kleines, vollständig lokal ausführbares German Small Language Model (SLM).  
Es wurde mit Python, PyTorch und einem eigenen BPE-Tokenizer trainiert.  

Das Projekt richtet sich an alle, die:

- ihr **eigenes KI-Modell** trainieren wollen  
- ein **vollständig offline laufendes SLM** suchen  
- verstehen möchten, wie Tokenizer, Training und Sampling zusammenarbeiten  
- eine erweiterbare Basis für Experimente oder Forschung brauchen  


---

# 🚀 Funktionsübersicht

| Funktion | Beschreibung |
|---------|--------------|
| **Eigener BPE-Tokenizer** | Keine externen Modelle notwendig – alles wird lokal trainiert. |
| **MiniGPT-Architektur** | Ein kleines GPT-Modell mit Embeddings, Self-Attention & Feedforward. |
| **Offline-Training** | Keine API, kein Internet, keine Cloud. |
| **Fortsetzbares Training** | Checkpoint-System (`checkpoint.pt`). |
| **2 GUI-Versionen** | V1 (Basic), V2 (Stil + Kontext + Reranking). |
| **Prompt-Stilspeicher (memory.json)** | Die KI passt Satzbau & Stil an deine Prompts an. |
| **Kontext-Modus (optional)** | Folgefragen erkennen („Wann erschien das Spiel?“). |
| **Temperatur, Top-K & Kandidaten-Reranking** | Voll kontrollierbare Textgenerierung. |

---

# 📁 Projektstruktur

```
sl-mai/
│
├── train.py                     # Training des Modells
├── model.py                     # MiniGPT Architektur
├── tokenizer.py                 # BPE-Tokenizer
├── data.py                      # Dataset / Block-Handling
├── gui_generate_V1.py           # Einfache GUI ohne Kontext
├── gui_generate_V2.py           # Erweiterte GUI (Stil, Satzbau, Kontext)
├── memory.py                    # Prompt-Speicher + Stilprofil
│── latest_training_files/
    |
    |── grundwissen.txt              # Deine Trainingsdaten (muss man selbst hinzufügen)
    |── tokenizer.json               # Nach Training erzeugt
    ├── checkpoint.pt                # Fortsetzbarer Trainingszustand
    ├── minigpt_grundwissen.pt       # Fertiges Modell
│
└── LICENSE                      # Lizenzbestimmungen
```

---

# 📘 1. Voraussetzungen

### 🐍 Python 3.9–3.12  
### 🔧 Module installieren:

```
pip install torch
```

und evtl.

```
pip install tkinter
```

(Unter Windows ist `tkinter` normalerweise bereits vorhanden.)

---

# 📘 2. Trainingsdaten: `grundwissen.txt`

Du musst im Projektordner eine Datei `grundwissen.txt` erstellen.

### Empfohlen:

- **UTF-8 Text**
- **deutsche, vollständige Sätze**
- **mind. 200 KB**, besser **800 KB – 2 MB**
- **Themenmix:** Wissenschaft, Erklärungen, Geschichte, Technologie, Q&A usw.
- Die KI lernt *nur* das, was hier drin steht  
  → je besser der Text, desto besser das Modell.

⚠️ **Wichtig:**  
Deine Trainingsdaten dürfen KEINE persönlichen Daten enthalten.  
Nur neutrale, allgemein gültige Texte verwenden.

---

# 🏋️ 3. Training starten

In den Projektordner wechseln:

```
cd sl-mai
```

Training ausführen:

```
python train.py
```

Du wirst gefragt:

```
Gib die Anzahl der Epochen an, die erreicht werden sollen:
Gib die Batches pro Epoche an:
```

### Beispiel (empfohlen):

```
80
250
```

### Während des Trainings werden erzeugt:

| Datei | Zweck |
|-------|-------|
| `tokenizer.json` | Dein Tokenizer |
| `checkpoint.pt` | Fortsetzbarer Trainingsstand |
| `minigpt_grundwissen.pt` | Das finale Modell |

Das Training kann jederzeit abgebrochen werden –  
beim nächsten Start wird automatisch fortgesetzt.

---

# 💬 4. Nutzung der GUIs

---

## 🎛️ **V1: Einfache GUI – keine Stillogik, kein Kontext**

```
python gui_generate_V1.py
```

Eigenschaften:

- beantwortet jede Frage separat  
- keine Stilübernahme  
- keine Prompt-Analyse  
- stabil & minimal

---

## 🎛️ **V2: Erweiterte GUI – Stil, Satzstruktur, Kontext**

```
python gui_generate_V2.py
```

### Funktionen:

#### 🟦 **Stil-Einfluss (Slider 0–1)**
Je höher der Wert, desto stärker orientiert sich die KI an:

- Satzbau deiner gespeicherten Prompts  
- Wortwahl  
- typischem Schreibstil  

#### 🟩 **Interne Kandidaten (1–10)**  
Das Modell erzeugt mehrere Rohvorschläge.  
Danach findet ein **Reranking** statt:

1. Grammatikpunkte  
2. Satzzeichen  
3. deutsche Wörter  
4. Wiederholungsstrafe  
5. Stilähnlichkeit  
6. Prompt-Ähnlichkeit  

→ Die **beste** Antwort wird angezeigt.

#### 🔥 **Temperatur (0.3–0.7)**  
- Niedrig → präzise, strikt, weniger kreativ  
- Hoch → kreativer, aber chaotischer  

#### 🟧 **Kontextmodus (Checkbox)**  
Wenn aktiviert, erkennt die KI einfache Folgefragen:

**Beispiel:**

Prompt 1:
> Warum ist Minecraft beliebt?

Prompt 2:
> Wann ist das Spiel erschienen?

→ Die KI weiß: „das Spiel“ = Minecraft.  
(Kommt auf Trainingsqualität + Prompt-Stil an.)

---

# 🧠 5. Wie die KI lernt (Wichtig!)

SL-MAI lernt **nicht** live aus Antworten.  
Er lernt aus zwei Dingen:

### 1. **Deinen Trainingsdaten (`grundwissen.txt`)**
– beeinflussen Wissen  
– beeinflussen Sprachqualität  
– beeinflussen Satzbau  
– verändern Gewichte → Training nötig  

### 2. **Deinen Prompts (memory.json)**
– beeinflussen Stil  
– beeinflussen Wortwahl  
– beeinflussen Satzstruktur  
– *kein Training nötig*  
– KI passt sich dynamisch an (Version V2)

---

# ⚠️ Einschränkungen & Hinweise

- SL-MAI ist ein **Mini-Modell**, kein GPT-4.  
- Es versteht Themen *oberflächlich*, abhängig vom Training.  
- Es erfindet gelegentlich Fakten („Halluzinationen“).  
- Kontext funktioniert nur in einfacher Form.  
- Sehr präzise Aufgaben übersteigen ein Mini-SLM.

---

# 📜 6. Lizenz (wichtiger Abschnitt)

Dieses Projekt ist **nicht-kommerziell**.  
Die Nutzung des Codes ist erlaubt, aber:

- Modelle dürfen **nicht kommerziell genutzt werden**  
- Trainingsdaten dürfen **nicht wiederverwendet oder weiterverkauft werden**  
- Der Name „Yenoyosh“ muss genannt werden  
- Die KI selbst darf **nicht als Dienst angeboten werden**

Siehe vollständige `LICENSE` im Repository.

---

# 💡 7. Beispiel-Prompt

```
Warum ist Photosynthese wichtig?
```

Beispielantwort (abhängig vom Training):

> Die Photosynthese ist wichtig, weil sie Pflanzen ermöglicht, Lichtenergie in chemische Energie umzuwandeln und gleichzeitig Sauerstoff produziert, der für viele Lebewesen lebensnotwendig ist.

---

# 🧩 8. Erweiterungen (optional)

Du kannst SL-MAI leicht erweitern:

- größere Modelle (mehr Layer, mehr Heads)  
- größere Trainingsdaten  
- Kontextfenster erhöhen (block_size z. B. 128)  
- grammatikbasierte Filter  
- POS-Tagger für echte Satzstrukturkontrolle  
- Reinforcement Learning für Stiloptimierung  

---

# 💖 9. Autor

**Yenoyosh**  
2025

---

# ✔️ Projektstatus

SL-MAI ist funktional, trainierbar und erweiterbar.  
Das Modell verbessert sich mit jeder Epoche und jeder Erweiterung.

Beiträge und Forks sind willkommen.
