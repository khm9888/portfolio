import numpy as np
import cv2

def showImage():
    imgfile = 'data\cleansing_data\sample.jpg'
    img = cv2.imread(imgfile,cv2.IMREAD_COLOR)
    cv2.imshow('model',img)
    cv2.waitKey(0)#입력값이 있을 때까지 기달린다. ()안의 값은 1000분의 1초
    cv2.destroyAllWindows()
    
showImage()