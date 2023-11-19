
import numpy as np
import cv2

img = cv2.imread("./pro2/opencv/images/output.jpg")
px = img[340,200]
print(px)
# img[340, 200] = [0, 0, 0]