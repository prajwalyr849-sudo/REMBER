"""
REMBER Risk Engine
Converts object detections into an interpretable risk score.

Input:
    A list of detections containing class_name, confidence,
    and optionally bbox = [x1, y1, x2, y2].

Output:
    Risk score (0-100), severity, contributing factors,
    and a human-readable explanation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional


# Configurable risk weights. These are initial demo values,
# not calibrated real-world safety probabilities.
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
    """Calculate an explainable risk score from detections."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        thresholds: Optional[dict[str, int]] = None,
    ):
        self.weights = {
            **DEFAULT_RISK_WEIGHTS,
            **(weights or {}),
        }
        self.thresholds = {
            **SEVERITY_THRESHOLDS,
            **(thresholds or {}),
        }
        self._validate_config()

    def _validate_config(self) -> None:
        for name, weight in self.weights.items():
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"Weight for {name!r} must be between 0 and 1"
                )

        values = list(self.thresholds.values())
        if any(not 0 <= value <= 100 for value in values):
            raise ValueError("Thresholds must be between 0 and 100")

        if values != sorted(values):
            raise ValueError("Thresholds must be in ascending order")

    @staticmethod
    def _get(detection: Any, key: str, default=None):
        """Support both dictionaries and detection objects."""
        if isinstance(detection, dict):
            return detection.get(key, default)
        return getattr(detection, key, default)

    @staticmethod
    def _normalise_name(value: Any) -> str:
        return str(value or "unknown").strip().lower()

    @staticmethod
    def _validate_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            raise ValueError("Detection confidence must be numeric")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        return confidence

    @staticmethod
    def _proximity_factor(
        bbox: Any,
        image_width: Optional[float],
        image_height: Optional[float],
    ) -> float:
        """
        Estimate visual prominence using bounding-box area.

        This is NOT physical distance. A large box may indicate
        an object close to the camera, but perspective can mislead.
        """
        if bbox is None or not image_width or not image_height:
            return 1.0

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")

        if len(bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")

        x1, y1, x2, y2 = map(float, bbox)

        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid bounding box coordinates")

        box_area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        area_ratio = min(1.0, max(0.0, box_area / image_area))

        # Factor ranges from 1.0 to 1.5.
        return 1.0 + 0.5 * area_ratio

    def score(
        self,
        detections: list[Any],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> RiskResult:
        """
        Score detections using a capped, cumulative risk model.

        Each detection contributes:
            class weight × confidence × prominence factor

        Contributions combine as independent risk contributions.
        """
        if not isinstance(detections, (list, tuple)):
            raise ValueError("detections must be a list")

        contributions = []
        factors = []

        for detection in detections:
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

            weight = float(self.weights.get(name, 0.10))

            if not 0 <= weight <= 1:
                raise ValueError(f"Invalid weight for {name!r}")

            bbox = self._get(detection, "bbox")
            proximity = self._proximity_factor(
                bbox, image_width, image_height
            )

            contribution = min(1.0, weight * confidence * proximity)
            contributions.append(contribution)

            if contribution > 0:
                factors.append({
                    "object": name,
                    "confidence": round(confidence, 3),
                    "weight": round(weight, 3),
                    "contribution": round(contribution, 4),
                })

        # Combine contributions without allowing the score
        # to exceed 100.
        remaining_safety = 1.0
        for contribution in contributions:
            remaining_safety *= 1.0 - contribution

        score = round((1.0 - remaining_safety) * 100)
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
                f"{item['object']} ({item['contribution'] * 100:.1f}%)"
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


# Convenient function for other REMBER modules.
def calculate_risk(
    detections: list[Any],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> dict[str, Any]:
    engine = RiskEngine()
    return engine.score(
        detections,
        image_width=image_width,
        image_height=image_height,
    ).to_dict()