from risk_engine.risk_score import calculate_risk

def test_low_risk():
    result = calculate_risk([
        {"label": "person", "confidence": 0.5}
    ])
    assert result["risk_level"] == "Low"


def test_medium_risk():
    result = calculate_risk([
        {"label": "fire", "confidence": 0.9}
    ])
    assert result["risk_level"] == "Medium"


def test_high_risk():
    result = calculate_risk([
        {"label": "fire", "confidence": 1.0},
        {"label": "accident", "confidence": 1.0}
    ])
    assert result["risk_level"] == "High"