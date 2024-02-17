import cv2 as cv 
import numpy as np


# open camera
while True:
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    # frame = cv.convertScaleAbs(frame, alpha=0.5, beta=0.2)
    # filter by bright neon orange
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([10, 255, 255])
    mask = cv.inRange(hsv, lower_orange, upper_orange)
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('k', frame)
    cv.imshow('frame', mask)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

