import cv2, numpy as np
"""
By franklintiel (Franklin Sarmiento)
- 13.4.2019
- franklinitiel@gmail.com
Example: Capture objects on real video with singular features.
Note: Objects with green colors are required.
Note: This tests have been applied using the camera integrated on the computer and using Ubuntu 17.10
"""

camera=cv2.VideoCapture(0)
kernel=np.ones((5,5), np.uint8)
while (True):
    received, frame = camera.read()
    if not received:
        break
    maxrange = np.array([255, 50, 255])
    minrange = np.array([51, 0, 51])
    mask = cv2.inRange(frame, minrange, maxrange)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    x,y,w,h = cv2.boundingRect(opening)
    cv2.rectangle(frame,(x,y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imshow('camera', frame)
    key=cv2.waitKey(1) & 0xFF
    if key == 27:
        break
