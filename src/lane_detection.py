import cv2

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

    # Show output
    cv2.imshow("Lane Detection - Step 1", blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()