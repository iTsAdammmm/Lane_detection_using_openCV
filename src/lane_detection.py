import cv2
import numpy as np

# ===============================
# Helper Functions
# ===============================

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


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
            continue

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


# ===============================
# Video Setup
# ===============================

video = cv2.VideoCapture("../assets/road.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "../assets/output_lane_detection.mp4",
    fourcc,
    fps,
    (width, height)
)

# ===============================
# Smoothing Variables
# ===============================

prev_left = None
prev_right = None
alpha = 0.85

# ===============================
# Main Loop
# ===============================

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # -------- Preprocessing --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    # -------- Region of Interest --------
    height, width = edges.shape[:2]
    roi_vertices = np.array([[
        (0, height),
        (width, height),
        (width // 2, int(height * 0.6))
    ]], dtype=np.int32)

    cropped_edges = region_of_interest(edges, roi_vertices)

    # -------- Hough Transform --------
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=60,
        maxLineGap=80
    )

    # -------- Lane Averaging --------
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

    # -------- Independent Smoothing --------
    if current_left is not None:
        if prev_left is None:
            prev_left = current_left
        else:
            prev_left = [
                int(alpha * p + (1 - alpha) * c)
                for p, c in zip(prev_left, current_left)
            ]

    if current_right is not None:
        if prev_right is None:
            prev_right = current_right
        else:
            prev_right = [
                int(alpha * p + (1 - alpha) * c)
                for p, c in zip(prev_right, current_right)
            ]

    # -------- Draw Lanes --------
    if prev_left is not None:
        cv2.line(
            lane_frame,
            (prev_left[0], prev_left[1]),
            (prev_left[2], prev_left[3]),
            (0, 255, 0),
            8
        )

    if prev_right is not None:
        cv2.line(
            lane_frame,
            (prev_right[0], prev_right[1]),
            (prev_right[2], prev_right[3]),
            (0, 255, 0),
            8
        )

    # -------- Lane Departure Warning --------
    if prev_left is not None and prev_right is not None:
        left_x_bottom = prev_left[0]
        right_x_bottom = prev_right[0]

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

    # -------- Display & Save --------
    cv2.imshow("Lane Detection with ADAS Warning", lane_frame)
    out.write(lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Cleanup
# ===============================

video.release()
out.release()
cv2.destroyAllWindows()