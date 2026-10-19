"""Explainability Engine generating human-interpretable rationales for recommendations."""
from typing import Dict, Any

class RecommendationExplainer:
    """Generates crisp, context-rich explanations explaining WHY each item was recommended."""

    @staticmethod
    def explain_movie(item: Dict[str, Any], mood_info: Dict[str, Any]) -> str:
        """Constructs a tailored one-line explanation for a recommended movie."""
        v = item.get("mood_valence", 0.5)
        e = item.get("mood_energy", 0.5)
        cf_r = item.get("cf_rating", 3.5)
        nov = item.get("novelty_score", 0.2)
        genres = ", ".join(item.get("genres", [])[:2])
        emotion = mood_info.get("dominant_emotion", "Neutral")
        intent = mood_info.get("intent", "balanced")

        # 1. High Novelty trigger
        if nov > 0.65:
            return f"Curated for high novelty: a celebrated hidden gem ({item['vote_average']}★) with {genres.lower()} storytelling matching your {emotion.lower()} state."

        # 2. Collaborative Filtering driven
        if cf_r >= 4.4:
            return f"High collaborative match: similar cinephiles rated this {cf_r}★, with {genres.lower()} pacing perfectly attuned to your {emotion.lower()} mood."

        # 3. Mood regulation / uplift transition
        if intent == "mood_regulation":
            return f"Guides your transition toward {emotion.lower()} with vibrant {genres.lower()} energy ({int(v * 100)}% positivity) and an uplifting resolution."

        # 4. Catharsis / low valence resonance
        if v < 0.4:
            return f"Deep emotional resonance: an evocative {genres.lower()} journey providing gentle catharsis and thoughtful atmosphere."

        # 5. High-energy hype / adrenaline
        if e > 0.8:
            return f"Matches your high-energy mindset with relentless {genres.lower()} pacing and exhilarating cinematic craft."

        # 6. Calm / cozy comfort
        if e < 0.4 and v > 0.6:
            return f"Comforting warm ambiance: a gentle {genres.lower()} narrative designed to soothe stress and help you unwind."

        return f"Aligned with your {emotion.lower()} mood through balanced {genres.lower()} themes and an impressive {item['vote_average']}★ reception."

    @staticmethod
    def explain_music(item: Dict[str, Any], mood_info: Dict[str, Any]) -> str:
        """Constructs a tailored one-line explanation for a recommended music track."""
        v = item.get("valence", 0.5)
        e = item.get("energy", 0.5)
        tempo = int(item.get("tempo", 110))
        dance = item.get("danceability", 0.5)
        nov = item.get("novelty_score", 0.2)
        genre = item.get("track_genre", "music")
        emotion = mood_info.get("dominant_emotion", "Neutral")
        intent = mood_info.get("intent", "balanced")

        # 1. High Novelty trigger
        if nov > 0.55:
            return f"Discovered for fresh variety: an atmospheric {genre} track matching your target {int(e * 100)}% energy curve."

        # 2. Mood regulation / uplifting shift
        if intent == "mood_regulation" or v > 0.85:
            return f"Picks you up with radiant {int(v * 100)}% harmonic valence and an infectious {tempo} BPM groove."

        # 3. Catharsis / slow melancholy / ambient
        if v < 0.35 and e < 0.35:
            return f"Meditative acoustic resonance: spacious instrumentation and quiet cadence ({tempo} BPM) to give you breathing room."

        # 4. High adrenaline / workout / anger release
        if e > 0.85:
            return f"High-octane sonic drive: {int(e * 100)}% raw energy and a driving {tempo} BPM tempo to fuel your momentum."

        # 5. Chill / cozy morning
        if e < 0.5 and v > 0.6:
            return f"Soothing {genre} texture: warm melodic warmth and gentle {tempo} BPM pacing ideal for unwinding."

        # 6. Dance / party groove
        if dance > 0.70:
            return f"Irresistible rhythmic bounce: {int(dance * 100)}% danceability and upbeat {genre} bassline fitting your {emotion.lower()} vibe."

        return f"Sonic profile mirrors your {emotion.lower()} frequency with {int(v * 100)}% valence and {tempo} BPM rhythm."
