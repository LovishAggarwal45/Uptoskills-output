import cv2
import numpy as np

def calculate_kinetic_agitation(prev_face, curr_face):
    """Calculates general optical flow velocity for sudden movements."""
    if prev_face is None: 
        return 0.0, curr_face
        
    flow = cv2.calcOpticalFlowFarneback(prev_face, curr_face, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    significant = magnitude[magnitude > 2.0]
    
    score = min(np.mean(significant) * 12.0, 100.0) if len(significant) > 0 else 0.0
    return float(score), curr_face

def calculate_vector_delta(prev_width, current_width):
    """Calculates micro-movement delta (e.g., sudden shouting or snarling)."""
    if prev_width is None: 
        return 0.0
        
    delta = abs(current_width - prev_width)
    return min(float(delta * 10.0), 100.0)