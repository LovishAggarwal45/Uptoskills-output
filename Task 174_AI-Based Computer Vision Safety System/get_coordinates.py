import cv2

# Mouse callback function to log clicked points
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selected Coordinate: X={x}, Y={y}")

# Read first frame of v1.mp4
cap = cv2.VideoCapture("data/v1.mp4")
success, frame = cap.read()

if success:
    print("Click on the top-left and bottom-right corners of the sink to get ROI points.")
    cv2.imshow("Select ROI Points - Press 'q' to close", frame)
    cv2.setMouseCallback("Select ROI Points - Press 'q' to close", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cap.release()