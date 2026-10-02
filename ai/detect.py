import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

def load_model():
    """LOAD THE PRETRAINED YOLO MODEL"""
    print("Loading REMBER AI model")
    model = YOLO(MODEL_PATH)
    print("Model loaded sucessfully")
    return model

def  detect_frame(frame,model,confidence=0.35):
    #detects objects in the video frame
    results = model.predict(
        source = frame,
        conf = confidence,
        verbose = False

    )
    result = results[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        detections.append({
            "class_id":class_id,
            "class_name":model.names[class_id],
            "confidence":round(score,3)
        })
    return result.plot(),detections

def main():
    model = load_model()
    print("Model type:",type(model))
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR!,Could not open webcam")
        return
    print("REMBER AI detection Started")
    print("Press Q to stop")

    try:
        while True:
            Sucess, frame = camera.read()
            if not Sucess:
                print("Could not read camera frame.")
                break
            annotated_frame, detections = detect_frame(frame,model)
            cv2.imshow("REMBER- AI Detection",annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
if __name__=="__main__":
    main( )
