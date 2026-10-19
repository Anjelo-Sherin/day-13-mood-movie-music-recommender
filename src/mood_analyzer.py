"""Mood and Emotion Analysis Engine based on Russell's Circumplex Model of Affect (Valence-Arousal Space)."""
import re
from typing import Dict, Any, Tuple

# Affective Lexicon mapping emotional keywords to (Valence, Arousal/Energy) in [0.0, 1.0]
AFFECTIVE_LEXICON = {
    # High Valence, High Energy (Joyful, Excited, Euphoric, Party)
    "happy": (0.85, 0.75, "Joy"),
    "joy": (0.90, 0.80, "Joy"),
    "joyful": (0.90, 0.80, "Joy"),
    "ecstatic": (0.95, 0.90, "Joy"),
    "cheerful": (0.85, 0.70, "Joy"),
    "excited": (0.88, 0.85, "Excitement"),
    "hyped": (0.85, 0.95, "Excitement"),
    "pumped": (0.85, 0.95, "Excitement"),
    "energetic": (0.80, 0.90, "Excitement"),
    "party": (0.85, 0.90, "Excitement"),
    "euphoric": (0.95, 0.88, "Joy"),
    "uplifting": (0.88, 0.75, "Uplift"),
    "uplifted": (0.88, 0.75, "Uplift"),
    "celebrate": (0.90, 0.85, "Joy"),
    "dancing": (0.85, 0.85, "Excitement"),
    "thrilled": (0.90, 0.88, "Excitement"),
    "fun": (0.85, 0.75, "Joy"),
    "exhilarated": (0.88, 0.92, "Excitement"),

    # High Valence, Low Energy (Calm, Peaceful, Cozy, Relaxed, Content)
    "calm": (0.80, 0.25, "Calm"),
    "peaceful": (0.85, 0.20, "Calm"),
    "relaxed": (0.80, 0.30, "Calm"),
    "chill": (0.75, 0.35, "Calm"),
    "cozy": (0.85, 0.30, "Calm"),
    "warm": (0.80, 0.35, "Calm"),
    "serene": (0.85, 0.20, "Calm"),
    "tranquil": (0.85, 0.18, "Calm"),
    "content": (0.78, 0.30, "Calm"),
    "grateful": (0.82, 0.35, "Calm"),
    "soothing": (0.82, 0.25, "Calm"),
    "gentle": (0.80, 0.28, "Calm"),
    "dreamy": (0.78, 0.32, "Calm"),
    "meditative": (0.75, 0.15, "Calm"),
    "mellow": (0.75, 0.30, "Calm"),

    # Romance / Love / Tender
    "romantic": (0.85, 0.45, "Romance"),
    "love": (0.88, 0.50, "Romance"),
    "tender": (0.80, 0.35, "Romance"),
    "in love": (0.90, 0.60, "Romance"),
    "sweet": (0.82, 0.40, "Romance"),

    # Low Valence, Low Energy (Sad, Melancholy, Lonely, Exhausted, Depressed)
    "sad": (0.20, 0.30, "Sadness"),
    "melancholy": (0.30, 0.28, "Melancholy"),
    "melancholic": (0.30, 0.28, "Melancholy"),
    "depressed": (0.15, 0.20, "Sadness"),
    "lonely": (0.22, 0.25, "Sadness"),
    "exhausted": (0.30, 0.18, "Exhaustion"),
    "tired": (0.35, 0.22, "Exhaustion"),
    "fatigued": (0.30, 0.18, "Exhaustion"),
    "drained": (0.25, 0.15, "Exhaustion"),
    "heartbroken": (0.15, 0.35, "Sadness"),
    "hopeless": (0.10, 0.20, "Sadness"),
    "somber": (0.28, 0.25, "Sadness"),
    "wistful": (0.42, 0.28, "Melancholy"),
    "nostalgic": (0.60, 0.35, "Nostalgia"),
    "pensive": (0.45, 0.28, "Melancholy"),
    "crying": (0.18, 0.40, "Sadness"),

    # Low Valence, High Energy (Angry, Stressed, Anxious, Furious, Tense)
    "angry": (0.18, 0.90, "Anger"),
    "furious": (0.12, 0.98, "Anger"),
    "mad": (0.20, 0.85, "Anger"),
    "stressed": (0.25, 0.80, "Stress"),
    "stress": (0.25, 0.80, "Stress"),
    "anxious": (0.22, 0.78, "Anxiety"),
    "nervous": (0.30, 0.72, "Anxiety"),
    "overwhelmed": (0.22, 0.82, "Stress"),
    "frustrated": (0.20, 0.85, "Anger"),
    "irritated": (0.25, 0.75, "Anger"),
    "tense": (0.25, 0.78, "Stress"),
    "panicked": (0.15, 0.95, "Anxiety"),
    "screaming": (0.18, 0.95, "Anger"),
    "hate": (0.15, 0.85, "Anger"),
    "dark": (0.30, 0.65, "Dark/Intense"),
    "intense": (0.40, 0.90, "Dark/Intense")
}

INTENSIFIERS = {
    "very": 1.25, "super": 1.35, "extremely": 1.45, "so": 1.25, "deeply": 1.3,
    "really": 1.2, "completely": 1.3, "insanely": 1.4, "immensely": 1.35
}

DIMINISHERS = {
    "a bit": 0.75, "slightly": 0.7, "somewhat": 0.8, "a little": 0.75, "mildly": 0.7
}

NEGATIONS = {"not", "never", "no", "hardly", "barely", "scarcely", "without"}

CONTRAST_CONJUNCTIONS = ["but", "however", "yet", "although", "though", "nonetheless", "still"]
DESIRE_KEYWORDS = ["want", "need", "wish", "looking for", "craving", "hoping for", "cheer me up", "unwind", "lift"]

class MoodAnalyzer:
    """Analyzes natural language mood sentences into valence, arousal (energy), and emotional trajectory."""

    def __init__(self):
        self.lexicon = AFFECTIVE_LEXICON

    def analyze(self, text: str) -> Dict[str, Any]:
        """Maps free-text mood to valence, energy, primary emotion, intent, and explanation."""
        clean_text = text.lower().strip()
        if not clean_text:
            return {
                "valence": 0.50,
                "energy": 0.50,
                "dominant_emotion": "Neutral",
                "intent": "balanced",
                "trajectory": "neutral exploration",
                "explanation": "No text provided; default neutral valence and energy applied."
            }

        # Check for contrastive structure (e.g. "exhausted but want something uplifting")
        target_clause, current_clause, has_contrast = self._split_clauses(clean_text)

        # Compute emotional vectors for both clauses
        cur_v, cur_e, cur_emotions = self._score_clause(current_clause)
        
        if has_contrast and target_clause:
            tgt_v, tgt_e, tgt_emotions = self._score_clause(target_clause)
            # The user wants to transition or regulate their state!
            final_valence = tgt_v
            # Energy target: if target explicitly asks for calm/chill, respect it; if uplifting, boost energy
            final_energy = tgt_e
            dominant_emotion = tgt_emotions[0][0] if tgt_emotions else "Uplift"
            intent = "mood_regulation"
            trajectory = f"From {cur_emotions[0][0] if cur_emotions else 'Current State'} -> {dominant_emotion}"
            explanation = (
                f"Detected contrastive mood: currently feeling {cur_emotions[0][0] if cur_emotions else 'low/fatigued'}, "
                f"with a clear desire to transition toward {dominant_emotion} (Target Valence: {final_valence:.2f}, Energy: {final_energy:.2f})."
            )
        else:
            final_valence = cur_v
            final_energy = cur_e
            dominant_emotion = cur_emotions[0][0] if cur_emotions else self._infer_quadrant_emotion(final_valence, final_energy)
            
            # Detect whether intent is catharsis vs amplification
            if final_valence < 0.4:
                intent = "catharsis"
                trajectory = f"Conforming to {dominant_emotion}"
                explanation = f"Detected {dominant_emotion.lower()} sentiment. Recommending deep, resonant, and empathetic content."
            elif final_energy > 0.7:
                intent = "amplification"
                trajectory = f"Amplifying {dominant_emotion}"
                explanation = f"Detected high-energy {dominant_emotion.lower()} mood. Recommending upbeat and high-tempo media."
            else:
                intent = "immersion"
                trajectory = f"Aligning with {dominant_emotion}"
                explanation = f"Detected {dominant_emotion.lower()} atmosphere with balanced energy and valence."

        return {
            "valence": float(round(final_valence, 3)),
            "energy": float(round(final_energy, 3)),
            "dominant_emotion": dominant_emotion,
            "intent": intent,
            "trajectory": trajectory,
            "explanation": explanation,
            "input_text": text
        }

    def _split_clauses(self, text: str) -> Tuple[str, str, bool]:
        """Splits sentences with contrastive or desire conjunctions (e.g., 'tired but want uplifting')."""
        for conj in CONTRAST_CONJUNCTIONS:
            pattern = rf"\b{conj}\b"
            if re.search(pattern, text):
                parts = re.split(pattern, text, maxsplit=1)
                return parts[1].strip(), parts[0].strip(), True

        for desire in DESIRE_KEYWORDS:
            pattern = rf"\b(and\s+)?{desire}\b"
            match = re.search(pattern, text)
            if match and match.start() > 5:
                current_part = text[:match.start()].strip()
                target_part = text[match.start():].strip()
                return target_part, current_part, True

        return "", text, False

    def _score_clause(self, clause: str) -> Tuple[float, float, list]:
        """Calculates valence, energy and emotion histogram for a specific text clause."""
        words = re.findall(r"\b[a-z'-]+\b", clause)
        matches = []
        emotion_counts = {}

        i = 0
        while i < len(words):
            word = words[i]
            # Check 2-word idioms (e.g. 'in love', 'a bit')
            two_word = f"{word} {words[i+1]}" if i + 1 < len(words) else ""
            matched_key = None
            if two_word in self.lexicon:
                matched_key = two_word
                step = 2
            elif word in self.lexicon:
                matched_key = word
                step = 1
            else:
                step = 1

            if matched_key:
                val, en, emotion = self.lexicon[matched_key]
                # Lookback for negations and intensifiers within 3 previous words
                multiplier = 1.0
                negated = False
                for prev in words[max(0, i - 3):i]:
                    if prev in NEGATIONS:
                        negated = True
                    if prev in INTENSIFIERS:
                        multiplier *= INTENSIFIERS[prev]
                    elif prev in DIMINISHERS:
                        multiplier *= DIMINISHERS[prev]

                if negated:
                    # Invert valence and dampen energy
                    val = 1.0 - val
                    en = max(0.2, en * 0.7)
                    emotion = "Opposite/Neutralized"

                matches.append((val, en, multiplier))
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            i += step

        if not matches:
            # Fallback heuristic: check positive/negative sentiment words
            return 0.50, 0.50, [("Neutral", 1)]

        total_weight = sum(m[2] for m in matches)
        avg_valence = sum(m[0] * m[2] for m in matches) / total_weight
        avg_energy = sum(m[1] * m[2] for m in matches) / total_weight

        # Normalize and clip
        avg_valence = max(0.05, min(0.98, avg_valence))
        avg_energy = max(0.05, min(0.98, avg_energy))

        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return avg_valence, avg_energy, sorted_emotions

    def _infer_quadrant_emotion(self, v: float, e: float) -> str:
        """Determines emotion label from Russell's 2D Circumplex coordinates."""
        if v >= 0.5 and e >= 0.5:
            return "Joy / Excitement"
        elif v >= 0.5 and e < 0.5:
            return "Calm / Peaceful"
        elif v < 0.5 and e >= 0.5:
            return "Intense / Tension"
        else:
            return "Melancholic / Somber"
