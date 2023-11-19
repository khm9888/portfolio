import numpy as np
import cv2
import matplotlib.pyplot as plt
def showImage():
    imgfile = 'data\cleansing_data\sample.jpg'
    img = cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.show()
    
showImage()