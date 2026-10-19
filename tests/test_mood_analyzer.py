"""Tests for Mood and Emotion Analysis Engine."""
import pytest
from src.mood_analyzer import MoodAnalyzer

@pytest.fixture
def analyzer():
    return MoodAnalyzer()

def test_contrastive_mood_resolution(analyzer):
    """User feels fatigued but wants uplifting media."""
    result = analyzer.analyze("exhausted after a long week but want something uplifting")
    assert result["intent"] == "mood_regulation"
    assert result["valence"] >= 0.75  # Target is uplifting
    assert result["dominant_emotion"] in ["Uplift", "Joy"]
    assert "Exhaustion" in result["explanation"]

def test_high_energy_hype(analyzer):
    """User feels energetic / hyped for workout or party."""
    result = analyzer.analyze("super hyped and excited for a late night gym session")
    assert result["energy"] >= 0.80
    assert result["valence"] >= 0.70
    assert result["dominant_emotion"] in ["Excitement", "Joy"]

def test_melancholy_catharsis(analyzer):
    """User feels sad/melancholic and seeks reflective music."""
    result = analyzer.analyze("feeling melancholic and lonely tonight")
    assert result["valence"] <= 0.40
    assert result["energy"] <= 0.45
    assert result["intent"] == "catharsis"

def test_negation_handling(analyzer):
    """Negations invert emotional polarities."""
    happy_res = analyzer.analyze("feeling cheerful")
    not_happy_res = analyzer.analyze("not cheerful today")
    assert happy_res["valence"] > not_happy_res["valence"]

def test_empty_string_neutral_fallback(analyzer):
    """Empty string returns balanced defaults."""
    res = analyzer.analyze("")
    assert res["valence"] == 0.50
    assert res["energy"] == 0.50
