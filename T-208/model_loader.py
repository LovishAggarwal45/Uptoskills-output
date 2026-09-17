import cv2
import os
import urllib.request

def get_yunet_model():
    """Downloads the modern ONNX YuNet Face Detection model."""
    model_name = "face_detection_yunet_2023mar.onnx"
    
    # Clean up corrupted downloads if they exist
    if os.path.isfile(model_name) and os.path.getsize(model_name) < 1000:
        os.remove(model_name)
        
    # Download the model securely
    if not os.path.isfile(model_name):
        print("Downloading modern ONNX Face Detection AI (YuNet)...")
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(model_name, 'wb') as out_file:
            out_file.write(response.read())
            
    return model_name

def initialize_detector(frame_width, frame_height, confidence):
    """Creates and returns the YuNet Face Detector object."""
    model_path = get_yunet_model()
    return cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(frame_width, frame_height),
        score_threshold=confidence
    )