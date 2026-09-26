from ultralytics import YOLO
import cv2


# ==========================================
# REMBER - AI Detection Module
# ==========================================

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")


def detect_frame(frame):
    """
    Detect objects in a single video frame.

    Returns:
        annotated_frame: Frame with bounding boxes
        detections: List of detected objects
    """

    # Run YOLO detection
    results = model(frame, verbose=False)
    result = results[0]

    detections = []

    # Extract detected objects
    for box in result.boxes:

        class_id = int(box.cls[0].item())
        label = model.names[class_id]

        confidence = float(box.conf[0].item())

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        # Store detection information
        detections.append({
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": [x1, y1, x2, y2]
        })

    # Draw bounding boxes on the frame
    annotated_frame = result.plot()

    return annotated_frame, detections


def main():

    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Try changing VideoCapture(0) to VideoCapture(1).")
        return

    print("===================================")
    print(" REMBER AI DETECTION STARTED")
    print(" Press Q to stop")
    print("===================================")

    frame_count = 0

    try:
        while True:

            # Read webcam frame
            success, frame = cap.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            # Detect objects in the frame
            annotated_frame, detections = detect_frame(frame)

            # Print detections every 30 frames
            if frame_count % 30 == 0:

                print("\nREMBER Detections:")

                if detections:
                    for detection in detections:
                        print(detection)
                else:
                    print("No objects detected.")

            # Display the annotated webcam feed
            cv2.imshow(
                "REMBER - AI Detection",
                annotated_frame
            )

            frame_count += 1

            # Press Q to stop detection
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDetection interrupted.")

    finally:
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

        print("REMBER AI Detection stopped.")


if __name__ == "__main__":
    main()