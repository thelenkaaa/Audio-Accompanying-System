from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")

def detect_and_track_objects(video_path, model, output_path="output_video.mp4"):
    """
    Detects and tracks objects in a video without skipping frames.
    
    :param video_path: str - Path to the input video file.
    :param model: object - Object detection model compatible with the given input.
    :param output_path: str - Path to save the processed video with bounding boxes.
    :return: dict - Dictionary containing detected objects and their time intervals.
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  

    detected_objects = {}  # Dictionary to store object appearance intervals
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        frame_number += 1  # Track current frame number

        # Detect objects
        results = model(frame)

        for result in results[0].boxes:  
            x1, y1, x2, y2 = result.xyxy[0].tolist()  
            conf = result.conf[0].item()  
            cls = result.cls[0].item()  
            class_name = model.names[int(cls)]  

            # Track object appearance times
            if class_name not in detected_objects:
                detected_objects[class_name] = []

            detected_objects[class_name].append(frame_number / fps)  # Store timestamps

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)  

    cap.release()
    out.release()

    # Convert detected frames to time intervals
    object_durations = {}
    for obj, times in detected_objects.items():
        start_time = times[0]
        duration = 0
        intervals = []

        for i in range(1, len(times)):
            if times[i] - times[i - 1] > (1 / fps):  # If gap detected, close interval
                intervals.append((start_time, times[i - 1]))
                start_time = times[i]
        intervals.append((start_time, times[-1]))  # Append last interval
        
        object_durations[obj] = intervals

    print("[INFO] Object screen times:")
    for obj, intervals in object_durations.items():
        for start, end in intervals:
            print(f"{obj}: {round(start, 2)}s - {round(end, 2)}s")

    return object_durations  # Return intervals of object appearances