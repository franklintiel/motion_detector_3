import time, cv2
"""
By franklintiel (Franklin Sarmiento)
- 13.4.2019
- franklinitiel@gmail.com
Motion detector Example (Standard and basic method)
Note: This tests have been applied using the camera integrated on the computer and using Ubuntu 17.10
"""


cam = cv2.VideoCapture(0)
back = None
while True:
    (recorded, frame) = cam.read()
    if not recorded:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if back is None:
        back = gray
        continue
    diff = cv2.absdiff(back, gray)
    threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations=2)
    contoursimg = threshold.copy()
    contours, hierarchy = cv2.findContours(contoursimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        (x, y, width, height) = cv2.boundingRect(c)
        rects.append([x, y, width, height])
    rects, weights = cv2.groupRectangles(rects, 1, 1.5)
    for points in rects:
        point1 = (points[0], points[1])
        point2 = (points[0]+points[2], points[1]+points[3])
        cv2.rectangle(frame, point1, point2, (50, 255, 255), 2)
    cv2.imshow("camera", frame)
    key = cv2.waitKey(1) & 0xFF
    time.sleep(0.005)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
