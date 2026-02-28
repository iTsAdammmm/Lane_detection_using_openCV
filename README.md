# Lane Detection using OpenCV

## What this project does

This project detects road lanes from a driving video using classical computer vision techniques.

It also:
- Smooths lane lines across frames
- Detects lane departure
- Saves the output video

---

## How it works

1. Convert frame to grayscale
2. Apply Gaussian blur
3. Detect edges using Canny
4. Apply Region of Interest
5. Detect lines using Hough Transform
6. Average left and right lanes
7. Apply smoothing
8. Show warning if vehicle deviates

---

## How to run

Install dependencies:

pip install opencv-python numpy

Then run:

python src/lane_detection.py

Output video will be saved in:
assets/output_lane_detection.mp4