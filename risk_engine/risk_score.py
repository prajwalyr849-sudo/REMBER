def calculate_risk(detections, incident_count=0):
    score = 0

    for detection in detections:
        label = detection["label"].lower()
        confidence = detection["confidence"]

        if label in ["fire", "smoke"]:
            score += 50 * confidence

        elif label in ["accident", "crash"]:
            score += 60 * confidence

        elif label == "flood":
            score += 50 * confidence

        elif label == "vehicle":
            score += 20 * confidence

        elif label == "person":
            score += 10 * confidence

    score += incident_count * 5
    score = min(score, 100)

    if score < 30:
        level = "Low"
    elif score < 70:
        level = "Medium"
    else:
        level = "High"

    return {
        "risk_score": round(score, 2),
        "risk_level": level
    }