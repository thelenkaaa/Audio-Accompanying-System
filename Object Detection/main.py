from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")

def detect_and_save_video(video_path, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) 

    detected_tags = [] 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        # Detect objects
        results = model(frame)

        # Loop through detections using the 'boxes' attribute of the Results object
        for result in results[0].boxes:  
            x1, y1, x2, y2 = result.xyxy[0].tolist()  
            conf = result.conf[0].item() 
            cls = result.cls[0].item()  
            class_name = model.names[int(cls)]  

            # Append the detected class name to the list
            detected_tags.append(class_name)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)  

    cap.release()
    out.release()
    
    # Print the list of detected tags
    print(f"[INFO] Detected Tags: {detected_tags}")
  
    unique_detected_tags = list(set(detected_tags))
    print(f"[INFO] Unique Detected Tags: {unique_detected_tags}")

    
    # Return the detected tags for use in further processing
    return unique_detected_tags


# video_path = "/content/drive/My Drive/video_test.mp4"  # Update with your path
# output_path = "highlighted_video.mp4"
# detected_tags = detect_and_save_video(video_path, output_path)