import cv2
import math
import config
from logger import initialize_logger, log_event
from model_loader import initialize_detector
from analytics import calculate_kinetic_agitation, calculate_vector_delta

def main():
    print("Initializing Modular Aggression Pre-Emption Monitor...")
    initialize_logger()
    
    # Setup Video
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"\nERROR: Could not open video file at '{config.VIDEO_PATH}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 
        
    out = cv2.VideoWriter(config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    detector = initialize_detector(frame_width, frame_height, config.CONFIDENCE_MIN)

    print(f"Processing video: {config.VIDEO_PATH}")
    face_history = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, faces = detector.detect(frame)
        current_faces = {}

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                x, y = max(0, x), max(0, y)
                w, h = min(frame_width - x, w), min(frame_height - y, h)
                if w <= 0 or h <= 0: continue

                # Extract FACS Vectors
                landmarks = face[4:14].astype(int)
                right_mouth, left_mouth = (landmarks[6], landmarks[7]), (landmarks[8], landmarks[9])
                mouth_width = math.hypot(left_mouth[0] - right_mouth[0], left_mouth[1] - right_mouth[1])

                # Match Face History
                face_center = (x + w//2, y + h//2)
                face_id = None
                for old_id, old_data in face_history.items():
                    if math.hypot(face_center[0] - old_data['center'][0], face_center[1] - old_data['center'][1]) < 50:
                        face_id = old_id
                        break
                
                prev_img = face_history[face_id]['img'] if face_id else None
                prev_mouth = face_history[face_id]['mouth_width'] if face_id else None
                face_id = face_id or f"face_{frame_count}_{x}"

                # Analyze Data
                standardized_face = cv2.resize(gray_frame[y:y+h, x:x+w], config.CROP_SIZE)
                kinetic_score, updated_img = calculate_kinetic_agitation(prev_img, standardized_face)
                facs_score = calculate_vector_delta(prev_mouth, mouth_width)
                
                total_risk = (kinetic_score * 0.4) + (facs_score * 0.6)
                current_faces[face_id] = {'img': updated_img, 'center': face_center, 'mouth_width': mouth_width}

                # Draw UI
                color = (0, 0, 255) if total_risk > config.THRESHOLD else (0, 255, 0)
                if total_risk > config.THRESHOLD:
                    log_event(config.LOCATION_ID, "Escalation Detected", total_risk)
                    cv2.putText(frame, "ESCALATION ALERT", (x, y - 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"Kinetic : {kinetic_score:.0f}", (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"FACS Vec: {facs_score:.0f}", (x + w + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, f"Total   : {total_risk:.0f}", (x + w + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        face_history = current_faces
        out.write(frame)

    cap.release()
    out.release() 
    print("Project compilation complete. Logs written to CSV.")

if __name__ == "__main__":
    main()