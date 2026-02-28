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

def make_points(y1,y2,line):
    slope, intercept = line
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return (x1,y1,x2,y2)

def average_slope_intercept(lines, height):
    left_fit = []
    right_fit = []

    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2-y1)/(x2-x1)
        intercept = y1-slope*x1
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    lanes = []
    y1=height
    y2= int(height*0.6)
    if left_fit:
        left_avg = np.average(left_fit,axis=0)
        lanes.append(make_points(y1,y2,left_avg))
    if right_fit:
        right_avg = np.average(right_fit,axis=0)
        lanes.append(make_points(y1,y2,right_avg))
    return lanes

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

    edges = cv2.Canny(blur, 75, 200)

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
        threshold=40,
        minLineLength=120,
        maxLineGap=30
    )

    lanes = average_slope_intercept(lines, frame.shape[0])
    lane_frame = frame.copy()
    if lanes is not None:
        for x1,y1,x2,y2 in lanes:
            cv2.line(lane_frame, (x1,y1), (x2,y2), (0,255,0), 8)

    # Show output
    cv2.imshow("Lane Detection - Lines", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()