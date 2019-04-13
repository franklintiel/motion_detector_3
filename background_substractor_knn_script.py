import time, cv2
"""
By franklintiel (Franklin Sarmiento)
- 13.4.2019
- franklinitiel@gmail.com
Motion detector Example (Using the algorithm "K-nearest neigbours")
Note: This tests have been applied using the camera integrated on the computer and using Ubuntu 17.10
"""


cam = cv2.VideoCapture(0)
bs = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)
cv2.ocl.setUseOpenCL(False)
while (True):
    recorded, frame = cam.read()
    if not recorded:
        break
    threshold = bs.apply(frame)
    contourimg = threshold.copy()
    contours, hierarchy = cv2.findContours(contourimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    key = cv2.waitKey(30) & 0xFF
    time.sleep(0.005)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
