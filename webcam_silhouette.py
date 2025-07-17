import cv2
import numpy as np

def create_silhouette():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Create background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    # Wait a moment for the camera to stabilize
    for i in range(30):
        _, _ = cap.read()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply background subtraction
        fgMask = backSub.apply(frame)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        
        # Create silhouette by masking original frame
        silhouette = frame.copy()
        silhouette[fgMask > 0] = [0, 0, 0]  # Set detected foreground to black
        
        # Show the result
        cv2.imshow('Silhouette', silhouette)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_silhouette()
