# 🔐 Sicherheitsrichtlinie

SL-MAI ist ein vollständig offline betriebenes, lokales Python-Modell.  
Das Projekt hat **keine Netzwerkfunktionen**, verarbeitet ausschließlich lokale Dateien  
und führt **keine externen Skripte oder fremden Befehle** aus.

Dadurch ist die Angriffsfläche sehr gering.

---

## 🛡️ Unterstützte Versionen

SL-MAI ist ein Hobby-/Forschungsprojekt.  
Unterstützung erfolgt **ohne Garantie und ohne feste Zeitfenster**.

| Version | Status |
|--------|--------|
| aktuelle GitHub-Version | ✔️ wird gelegentlich gepflegt |
| ältere Versionen | ❌ keine Updates |

---

## 🧪 Mögliche Sicherheitsprobleme

Obwohl SL-MAI offline arbeitet, können theoretisch folgende Probleme auftreten:

- Abstürze durch extrem große Eingaben  
- Fehler beim Lesen beschädigter Dateien  
- Unerwartetes Verhalten bei manipulierten JSON-/TXT-Dateien  

Es bestehen jedoch **keine externen Sicherheitsrisiken**, da das Projekt:

- keine Internetverbindungen herstellt  
- keine Befehle außerhalb des Python-Prozesses ausführt  
- keine Benutzerrechte ändert  
- keine sensiblen Systemfunktionen nutzt

---

## 📢 Melden von Problemen

Wenn du ein Problem findest (Fehler, unerwartetes Verhalten, mögliche Schwachstelle):

→ **Erstelle ein Issue im GitHub-Repository.**

Du musst keine privaten Daten angeben.

---

## ⚠️ Haftungsausschluss

SL-MAI wird im Ist-Zustand („as is“) bereitgestellt.  
Es gibt **keine Garantie** für Sicherheit, Support oder Aktualität.

Die Nutzung erfolgt **auf eigene Verantwortung**.

