#Runs object detection using camera and YOLO model
from risk_engine.risk_score import calculate_risk

import cv2
from ultralytics import YOLO  #you only look once

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
    accepted_detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        detections.append({
            "class_id":class_id,
            "class_name":model.names[class_id],
            "confidence":round(score,3)
        })
    
    for detection in detections:
        if detection["confidence"] > 0.70:
            accepted_detections.append(detection)
        
    return result.plot(),accepted_detections

def main():
    model = load_model()            #loads YOLO
    print("Model type:",type(model))
    camera = cv2.VideoCapture(0)        #opens default camera
    if not camera.isOpened():           #checks the status of camera
        print("ERROR!,Could not open webcam")
        return
    print("REMBER AI detection Started")
    print("Press Q to stop")
    frame_count = 0
    try:
        while True:
        
            Sucess, frame = camera.read()        #sucess tells whether frame reads sucessfully , frame contains image data 
            frame_count += 1           
            if not Sucess:
                print("Could not read camera frame.")
                break
            annotated_frame, accepted_detections = detect_frame(frame,model)
            cv2.imshow("REMBER- AI Detection",annotated_frame)      # displays camera image with detected objects  
            if frame_count % 30 == 0:
                
                risk_result = calculate_risk(accepted_detections)
                print("RISK RESULT:",risk_result)
                
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()            #releases camera 
        cv2.destroyAllWindows()     #closes opencv
if __name__=="__main__":
    main( )
