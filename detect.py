import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Define the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set the previous frame
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for optical flow calculation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h > 500: # filter out small bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate optical flow for the current object
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w], None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Calculate the average flow for the current object
                avg_flow = np.mean(flow, axis=(0, 1))

                # Predict the future position of the object
                future_x = int(x + avg_flow[0] * 10)
                future_y = int(y + avg_flow[1] * 10)

                # Draw a red bounding box around the future position
                cv2.rectangle(frame, (future_x, future_y), (future_x + w, future_y + h), (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    prev_frame = frame
    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

