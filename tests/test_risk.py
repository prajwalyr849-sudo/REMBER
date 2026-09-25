import pytest

from risk_engine.risk_score import (
    RiskEngine,
    calculate_risk,
)


@pytest.fixture
def engine():
    return RiskEngine()


def test_empty_detections_are_low_risk(engine):
    result = engine.score([])

    assert result.score == 0
    assert result.severity == "low"
    assert result.factors == []


def test_high_risk_detection(engine):
    result = engine.score([
        {"class_name": "fire", "confidence": 0.99}
    ])

    assert result.score >= 85
    assert result.severity == "critical"


def test_unknown_class_uses_default_weight(engine):
    result = engine.score([
        {"class_name": "unknown_object", "confidence": 1.0}
    ])

    assert result.score == 10
    assert result.severity == "low"


def test_confidence_affects_score(engine):
    low = engine.score([
        {"class_name": "fire", "confidence": 0.2}
    ])
    high = engine.score([
        {"class_name": "fire", "confidence": 0.9}
    ])

    assert high.score > low.score


def test_multiple_detections_increase_risk(engine):
    one = engine.score([
        {"class_name": "person", "confidence": 0.9}
    ])
    two = engine.score([
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "person", "confidence": 0.9},
    ])

    assert two.score > one.score


def test_bbox_changes_prominence(engine):
    small = engine.score(
        [{"class_name": "person", "confidence": 1.0,
          "bbox": [0, 0, 10, 10]}],
        image_width=100,
        image_height=100,
    )
    large = engine.score(
        [{"class_name": "person", "confidence": 1.0,
          "bbox": [0, 0, 90, 90]}],
        image_width=100,
        image_height=100,
    )

    assert large.score > small.score


def test_invalid_confidence_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.score([
            {"class_name": "person", "confidence": 1.5}
        ])


def test_invalid_bbox_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.score(
            [{"class_name": "person", "confidence": 0.9,
              "bbox": [10, 10, 5, 5]}],
            image_width=100,
            image_height=100,
        )


def test_result_can_be_converted_to_dict():
    result = calculate_risk([
        {"class_name": "car", "confidence": 0.9}
    ])

    assert isinstance(result, dict)
    assert 0 <= result["score"] <= 100
    assert result["severity"] in {
        "low", "moderate", "high", "critical"
    }
    assert "explanation" in result
import json


def test_custom_risk_weights():
    engine = RiskEngine(weights={"fire": 0.50})

    result = engine.score([
        {"class_name": "fire", "confidence": 1.0}
    ])

    assert result.score == 50
    assert result.severity == "moderate"

def test_severity_threshold_boundaries():
    engine = RiskEngine()

    assert engine._severity(19) == "low"
    assert engine._severity(20) == "low"

    assert engine._severity(44) == "low"
    assert engine._severity(45) == "moderate"

    assert engine._severity(69) == "moderate"
    assert engine._severity(70) == "high"

    assert engine._severity(84) == "high"
    assert engine._severity(85) == "critical"
