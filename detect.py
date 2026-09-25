from ultralytics import YOLO
import cv2

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("REMBER AI Detection started.")
print("Press Q to stop.")

frame_count = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        # Run object detection
        results = model(frame, verbose=False)
        result = results[0]

        detections = []

        # Process each detected object
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = model.names[class_id]
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2]
            })

        # Print results every 30 frames
        if frame_count % 30 == 0:
            print("\nREMBER Detections:")

            if detections:
                for detection in detections:
                    print(detection)
            else:
                print("No objects detected.")

        # Display bounding boxes
        annotated_frame = result.plot()

        cv2.imshow(
            "REMBER - AI Detection",
            annotated_frame
        )

        frame_count += 1

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("REMBER AI Detection stopped.")