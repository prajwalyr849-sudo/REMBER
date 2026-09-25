"""
REMBER Risk Engine.

Converts object detections into an interpretable risk score.
Weights and thresholds are configurable demo parameters,
not calibrated probabilities of real-world danger.
"""

from dataclasses import dataclass, asdict
from math import isfinite
from typing import Any, Optional


DEFAULT_RISK_WEIGHTS = {
    "person": 0.25,
    "bicycle": 0.25,
    "motorcycle": 0.50,
    "car": 0.45,
    "bus": 0.55,
    "truck": 0.60,
    "train": 0.60,
    "fire": 0.95,
    "smoke": 0.80,
    "knife": 0.85,
    "gun": 0.95,
    "weapon": 0.95,
}

SEVERITY_THRESHOLDS = {
    "low": 20,
    "moderate": 45,
    "high": 70,
    "critical": 85,
}


@dataclass
class RiskResult:
    score: int
    severity: str
    factors: list[dict[str, Any]]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskEngine:
    """Calculate explainable risk scores from detections."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        thresholds: Optional[dict[str, int]] = None,
    ):
        if weights is not None and not isinstance(weights, dict):
            raise ValueError("weights must be a dictionary")

        if thresholds is not None and not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a dictionary")

        self.weights = dict(DEFAULT_RISK_WEIGHTS)
        for name, weight in (weights or {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Weight names must be non-empty strings")
            self.weights[name.strip().lower()] = weight

        self.thresholds = dict(SEVERITY_THRESHOLDS)
        self.thresholds.update(thresholds or {})

        self._validate_config()

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be numeric") from None

        if not isfinite(number):
            raise ValueError(f"{name} must be finite")

        return number

    def _validate_config(self) -> None:
        for name, weight in self.weights.items():
            value = self._finite_number(weight, f"Weight for {name!r}")
            if not 0 <= value <= 1:
                raise ValueError(
                    f"Weight for {name!r} must be between 0 and 1"
                )
            self.weights[name] = value

        expected = {"low", "moderate", "high", "critical"}
        if set(self.thresholds) != expected:
            raise ValueError(
                f"Thresholds must contain exactly: {sorted(expected)}"
            )

        for name in expected:
            value = self.thresholds[name]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Threshold {name!r} must be an integer")

            if not 0 <= value <= 100:
                raise ValueError("Thresholds must be between 0 and 100")

        ordered = [
            self.thresholds[name]
            for name in ("low", "moderate", "high", "critical")
        ]

        if ordered != sorted(ordered):
            raise ValueError("Thresholds must be in ascending order")

    @staticmethod
    def _get(detection: Any, key: str, default=None):
        if isinstance(detection, dict):
            return detection.get(key, default)
        return getattr(detection, key, default)

    @staticmethod
    def _normalise_name(value: Any) -> str:
        if value is None:
            return "unknown"
        return str(value).strip().lower() or "unknown"

    @classmethod
    def _validate_confidence(cls, value: Any) -> float:
        confidence = cls._finite_number(value, "Confidence")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        return confidence

    @classmethod
    def _proximity_factor(
        cls,
        bbox: Any,
        image_width: Optional[float],
        image_height: Optional[float],
    ) -> float:
        """Estimate visual prominence, not physical distance."""

        if bbox is None:
            return 1.0

        if image_width is None or image_height is None:
            raise ValueError(
                "Both image_width and image_height are required with bbox"
            )

        width = cls._finite_number(image_width, "image_width")
        height = cls._finite_number(image_height, "image_height")

        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")

        try:
            if len(bbox) != 4:
                raise ValueError
            x1, y1, x2, y2 = [
                cls._finite_number(v, "Bounding box coordinate")
                for v in bbox
            ]
        except (TypeError, ValueError):
            raise ValueError(
                "bbox must contain four finite numeric coordinates"
            ) from None

        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid bounding box coordinates")

        # Clip the box to the image bounds.
        x1 = min(width, max(0.0, x1))
        x2 = min(width, max(0.0, x2))
        y1 = min(height, max(0.0, y1))
        y2 = min(height, max(0.0, y2))

        area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)

        return 1.0 + 0.5 * min(1.0, area_ratio)

    def score(
        self,
        detections: list[Any],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> RiskResult:
        if not isinstance(detections, (list, tuple)):
            raise ValueError("detections must be a list")

        contributions = []
        factors = []

        for detection in detections:
            if not isinstance(detection, dict) and detection is None:
                raise ValueError("Detection cannot be None")

            name = self._normalise_name(
                self._get(
                    detection,
                    "class_name",
                    self._get(detection, "label", "unknown"),
                )
            )

            confidence = self._validate_confidence(
                self._get(detection, "confidence", 1.0)
            )

            weight = self.weights.get(name, 0.10)
            bbox = self._get(detection, "bbox")

            prominence = self._proximity_factor(
                bbox, image_width, image_height
            )

            contribution = min(
                1.0, weight * confidence * prominence
            )
            contributions.append(contribution)

            if contribution > 0:
                factors.append({
                    "object": name,
                    "confidence": round(confidence, 3),
                    "weight": round(weight, 3),
                    "contribution": round(contribution, 4),
                })

        remaining = 1.0
        for contribution in contributions:
            remaining *= 1.0 - contribution

        score = round((1.0 - remaining) * 100)
        score = max(0, min(100, score))

        severity = self._severity(score)

        if not factors:
            explanation = "No risk contribution detected."
        else:
            top_factors = sorted(
                factors,
                key=lambda item: item["contribution"],
                reverse=True,
            )[:3]

            descriptions = [
                f"{item['object']} "
                f"({item['contribution'] * 100:.1f}%)"
                for item in top_factors
            ]

            explanation = (
                f"{len(factors)} contributing detection(s). "
                f"Main factors: {', '.join(descriptions)}."
            )

        return RiskResult(
            score=score,
            severity=severity,
            factors=factors,
            explanation=explanation,
        )

    def _severity(self, score: int) -> str:
        if score >= self.thresholds["critical"]:
            return "critical"
        if score >= self.thresholds["high"]:
            return "high"
        if score >= self.thresholds["moderate"]:
            return "moderate"
        return "low"


def calculate_risk(
    detections: list[Any],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> dict[str, Any]:
    """Convenience function returning a JSON-compatible dictionary."""
    return RiskEngine().score(
        detections,
        image_width=image_width,
        image_height=image_height,
    ).to_dict()