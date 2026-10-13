import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# 1. Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# 2. Dataset video input
video_path = "data\\Avenue_Dataset\\Avenue Dataset\\testing_videos\\1.mp4"  # <-- Update this to your file path
cap = cv2.VideoCapture(video_path)

# Parameters
CROWD_THRESHOLD = 6  # Minimum number of people considered an abnormal gathering
LOCATION_TAG = "Zone-A_Entrance"
records = []

# Optional: Video writer to save the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
out = cv2.VideoWriter('output_crowd_analysis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print("Processing video... Press 'q' on the video window to stop early.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on 'person' class (class 0 in COCO)
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    
    # Get bounding boxes
    boxes = results[0].boxes
    person_count = len(boxes) if boxes is not None else 0

    # Determine condition and risk score (scale 0.0 to 1.0)
    is_abnormal = person_count >= CROWD_THRESHOLD
    condition = "Abnormal Crowd Gathering" if is_abnormal else "Normal"
    risk_score = round(min(1.0, person_count / (CROWD_THRESHOLD * 1.5)), 2)

    # Annotate frame
    annotated_frame = results[0].plot()
    
    # Visual banner
    status_color = (0, 0, 255) if is_abnormal else (0, 255, 0) # Red for alert, Green for normal
    cv2.rectangle(annotated_frame, (10, 10), (480, 80), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Density: {person_count} persons", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Status: {condition}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if is_abnormal:
        cv2.putText(annotated_frame, "ALERT: CROWD GATHERING DETECTED", (frame_width // 12, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
    # --- MOVED OUTSIDE THE IF STATEMENT ---
    # Now it logs the data for every single frame processed
    records.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": LOCATION_TAG,
        "detected_condition": condition,
        "confidence_risk_score": risk_score
    })

    # Display & write frame
    out.write(annotated_frame)
    cv2.imshow("Crowd Surveillance System", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 3. Export CSV report
if records:
   # 3. Export CSV report
    df = pd.DataFrame(records)
    # Drop duplicates within same second to keep CSV clean
    df = df.drop_duplicates(subset=['timestamp'])
    df.to_csv("crowd_report.csv", index=False)
    print("\nProcessing complete! 'crowd_report.csv' and 'output_crowd_analysis.mp4' have been generated.")
else:
    print("\nProcessing complete! No abnormal crowd gatherings detected.")