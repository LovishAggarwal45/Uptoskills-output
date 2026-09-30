import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

# 1. Coordinates & Setup
ROI_X1, ROI_Y1 = 292, 555  
ROI_X2, ROI_Y2 = 479, 783  
MIRROR_X1, MIRROR_Y1 = 290, 150  
MIRROR_X2, MIRROR_Y2 = 480, 500  
EXIT_LINE_Y = 850 
STATION_ID = "Hallway_Station_01"

# --- NEW: Create Directory for Evidence Snapshots ---
os.makedirs("evidence", exist_ok=True)

csv_filename = "compliance_report.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Station ID", "Person ID", "Compliance Status"])

DATA_DIR = "data"
video_files = ["v1.mp4", "v2.mp4", "v3.mp4", "v4.mp4"]
model = YOLO("yolov8n-pose.pt")

person_states = {}

for video_name in video_files:
    video_path = os.path.join(DATA_DIR, video_name)
    if not os.path.exists(video_path):
        continue

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- NEW: Make a clean copy of the frame for the evidence snapshot ---
        # This ensures our saved images don't have messy bounding boxes drawn all over them
        clean_frame = frame.copy()
        
        # Draw Zones on the live display frame
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (MIRROR_X1, MIRROR_Y1), (MIRROR_X2, MIRROR_Y2), (128, 128, 128), 2)
        cv2.line(frame, (0, EXIT_LINE_Y), (frame.shape[1], EXIT_LINE_Y), (0, 0, 255), 2)

        results = model.track(frame, persist=True)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            keypoints = results[0].keypoints.xy.cpu().numpy()

            for box, track_id, kpts in zip(boxes, track_ids, keypoints):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Ignore Mirror Reflections
                if MIRROR_X1 < center_x < MIRROR_X2 and MIRROR_Y1 < center_y < MIRROR_Y2:
                    continue 

                if track_id not in person_states:
                    person_states[track_id] = {"washed": False, "logged": False}

                in_zone = False
                
                if len(kpts) > 10:
                    # Hand Tracking Logic
                    left_wrist, right_wrist = kpts[9], kpts[10]
                    for wrist in [left_wrist, right_wrist]:
                        wx, wy = int(wrist[0]), int(wrist[1])
                        if wx > 0 and wy > 0: 
                            if ROI_X1 < wx < ROI_X2 and ROI_Y1 < wy < ROI_Y2:
                                in_zone = True
                                person_states[track_id]["washed"] = True
                                cv2.circle(frame, (wx, wy), 8, (0, 255, 255), -1)
                    
                    # --- NEW: Face Blurring Logic ---
                    # YOLOv8 Pose uses keypoints 0-4 for facial features (nose, eyes, ears)
                    face_kpts = [kpts[i] for i in range(5) if kpts[i][0] > 0 and kpts[i][1] > 0]
                    
                    if face_kpts:
                        # Create a bounding box around the facial keypoints with a padding margin
                        fx_min = max(0, int(min([k[0] for k in face_kpts])) - 30)
                        fy_min = max(0, int(min([k[1] for k in face_kpts])) - 40)
                        fx_max = min(clean_frame.shape[1], int(max([k[0] for k in face_kpts])) + 30)
                        fy_max = min(clean_frame.shape[0], int(max([k[1] for k in face_kpts])) + 40)
                        
                        # Apply a strong Gaussian blur to the face region on the clean frame
                        face_region = clean_frame[fy_min:fy_max, fx_min:fx_max]
                        if face_region.size > 0:
                            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                            clean_frame[fy_min:fy_max, fx_min:fx_max] = blurred_face

                # --- NEW: Exit Logic & Evidence Snapshot ---
                if center_y > EXIT_LINE_Y and not person_states[track_id]["logged"]:
                    status = "Compliant" if person_states[track_id]["washed"] else "NON-COMPLIANT"
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 1. Write to CSV
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([current_time, STATION_ID, track_id, status])
                    
                    # 2. Save Privacy-Aware Snapshot
                    if status == "NON-COMPLIANT":
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        evidence_filename = f"evidence/non_compliant_ID{track_id}_{timestamp_str}.jpg"
                        # Save the 'clean_frame' which has the blurred face but no UI boxes
                        cv2.imwrite(evidence_filename, clean_frame)
                        print(f"Alert! Saved Privacy Evidence: {evidence_filename}")
                    
                    person_states[track_id]["logged"] = True
                    print(f"Logged ID {track_id}: {status}")

                # Display logic
                color = (0, 255, 0) if in_zone else (0, 0, 255) 
                if person_states[track_id]["washed"]:
                    status_text = "Status: PASS"
                    color = (0, 255, 0)
                else:
                    status_text = "Status: PENDING"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id} {status_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Hand Hygiene Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

cv2.destroyAllWindows()