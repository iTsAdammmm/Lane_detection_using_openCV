import cv2
import numpy as np

# -----------------------------
# Helper Functions
# -----------------------------

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def make_points(y1, y2, line):
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)


def average_slope_intercept(lines, height):
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            continue  # avoid division by zero

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lanes = []
    y1 = height
    y2 = int(height * 0.6)

    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        lanes.append(make_points(y1, y2, left_avg))

    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        lanes.append(make_points(y1, y2, right_avg))

    return lanes


# -----------------------------
# Main Program
# -----------------------------

prev_left = None
prev_right = None
alpha = 0.85
video = cv2.VideoCapture("../assets/road.mp4")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 75, 200)

    # Define Region of Interest (triangle)
    height, width = edges.shape[:2]
    roi_vertices = np.array([[
        (0, height),
        (width, height),
        (width // 2, int(height * 0.6))
    ]], dtype=np.int32)

    cropped_edges = region_of_interest(edges, roi_vertices)

    # Hough Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=120,
        maxLineGap=30
    )

    # Lane Detection and Smoothing
    lanes = average_slope_intercept(lines, frame.shape[0])

    lane_frame = frame.copy()

    current_left = None
    current_right = None

    if lanes is not None:
        for lane in lanes:
            x1, y1, x2, y2 = lane
            slope = (y2 - y1) / (x2 - x1)

            if slope < 0:
                current_left = lane
            else:
                current_right = lane

    # ---- Smooth Left Lane ----
    if current_left is not None:
        if prev_left is None:
            prev_left = current_left
        else:
            prev_left = [
                int(alpha * p + (1 - alpha) * c)
                for p, c in zip(prev_left, current_left)
            ]

    # ---- Smooth Right Lane ----
    if current_right is not None:
        if prev_right is None:
            prev_right = current_right
        else:
            prev_right = [
                int(alpha * p + (1 - alpha) * c)
                for p, c in zip(prev_right, current_right)
            ]

    # ---- Draw Lanes ----
    if prev_left is not None:
        cv2.line(lane_frame,
                (prev_left[0], prev_left[1]),
                (prev_left[2], prev_left[3]),
                (0, 255, 0), 8)

    if prev_right is not None:
        cv2.line(lane_frame,
                (prev_right[0], prev_right[1]),
                (prev_right[2], prev_right[3]),
                (0, 255, 0), 8)
        
    # -----------------------------
    # Lane Departure Warning
    # -----------------------------
    if lanes is not None and len(lanes) == 2:
        left_lane = lanes[0]
        right_lane = lanes[1]

        left_x_bottom = left_lane[0]
        right_x_bottom = right_lane[0]

        lane_center = (left_x_bottom + right_x_bottom) // 2
        frame_center = frame.shape[1] // 2

        deviation = frame_center - lane_center

        if abs(deviation) > 50:
            cv2.putText(
                lane_frame,
                "LANE DEPARTURE WARNING",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    cv2.imshow("Lane Detection with ADAS Warning", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()