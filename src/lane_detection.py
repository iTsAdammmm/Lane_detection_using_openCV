import cv2
import numpy as np

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    roi = cv2.bitwise_and(img,mask)
    return roi

def draw_lines(img, lines, color=(0,255,0), thickness=5):
    if lines is None:
        return img
    line_img = np.zeros_like(img)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1,y1), (x2,y2), color, thickness)
    
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

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

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho = 1,
        theta = np.pi / 180,
        threshold = 50,
        minLineLength = 100,
        maxLineGap = 50
    )

    lane_frame = draw_lines(frame, lines)

    # Show output
    cv2.imshow("Lane Detection - Lines", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()