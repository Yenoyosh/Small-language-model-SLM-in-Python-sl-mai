# context_manager.py

class ContextManager:
    """
    Kleiner Kontext-Manager nur für die laufende GUI-Sitzung.
    - Speichert letzte (Prompt, Antwort)-Paare.
    - Kann Folgefragen mit dem letzten Kontext anreichern.
    - Schreibt NICHTS auf die Festplatte und ändert kein Training.
    """

    def __init__(self, max_history: int = 10):
        self.history = []  # Liste von {"prompt": ..., "answer": ...}
        self.max_history = max_history

    def reset(self):
        """Kompletten Verlauf verwerfen (neuer Dialog)."""
        self.history.clear()

    def update(self, prompt: str, answer: str, enabled: bool = True):
        """Neues Prompt/Antwort-Paar speichern, falls aktiviert."""
        if not enabled:
            return
        self.history.append({"prompt": prompt, "answer": answer})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def apply(self, prompt: str, enabled: bool = True) -> str:
        """
        Gibt entweder den Prompt unverändert zurück,
        oder reichert ihn mit dem letzten Kontext an.

        Ziel: Folgefragen wie „Und was bedeutet das?“, „Warum?“ etc.
        bekommen den vorherigen Dialog mit dazu.
        Neue, unabhängige Fragen sollen möglichst unverändert bleiben.
        """
        if not enabled or not self.history:
            return prompt

        last = self.history[-1]
        last_prompt = last["prompt"].strip()
        last_answer = last["answer"].strip()

        # Heuristik: kurze/anschließende Fragen → Kontext davor
        low = prompt.lower()
        is_follow_up = False

        # einfache Trigger-Wörter für Folgefragen
        follow_words = [
            "und was", "und wie", "und warum", "warum", "wieso",
            "wie genau", "was bedeutet das", "und das", "und dann",
            "nochmal", "erkläre das", "erklär das", "was heißt das",
        ]
        for w in follow_words:
            if w in low:
                is_follow_up = True
                break

        # sehr kurze Prompts ohne Thema → wahrscheinlich Anschlussfrage
        if len(prompt.split()) <= 4:
            is_follow_up = True

        if not is_follow_up:
            # unabhängiger Prompt → unverändert zurück
            return prompt

        # Kontextpräfix bauen
        context_prefix = (
            "Vorheriger Kontext:\n"
            f"Benutzer: {last_prompt}\n"
            f"KI: {last_answer}\n\n"
            "Neue Frage:\n"
        )
        return context_prefix + prompt
