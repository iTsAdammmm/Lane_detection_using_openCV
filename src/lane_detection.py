import cv2
import numpy as np

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    roi = cv2.bitwise_and(img,mask)
    return roi

# Load video
video = cv2.VideoCapture("../assets/road.mp4")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Add fps display
    fps = int(video.get(cv2.CAP_PROP_FPS))
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur,50,150)
    height, width = edges.shape[:2]
    roi_vertices = np.array([[
        (0,height),
        (width, height),
        (width // 2, height // 2)
    ]], dtype=np.int32)
    cropped_edges = region_of_interest(edges, roi_vertices)

    # Show output
    cv2.imshow("Lane Detection - ROI Edges", cropped_edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()