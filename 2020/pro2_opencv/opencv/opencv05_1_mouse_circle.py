# 우리가 구현할 코드는 아래의 기능을 가지고 있습니다.

# 512 x 512 크기의 검정색 그림판 위에 마우스를 더블클릭하면 그곳에 반지름이 50 pixel인 원을 그림
# 원을 채우는 색상은 무작위로 선택됨
# ESC 키를 누르면 프로그램을 종료함

import numpy as np
import cv2

from random import shuffle

b = [i for i in range(256)]
g = [i for i in range(256)]
r = [i for i in range(256)]

def onMouse(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDBLCLK:
        shuffle(b), shuffle(g), shuffle(r)
        cv2.circle(param, (x,y), 50, (b[0], g[0], r[0]), -1)

def mouseBrush():
    img = np.zeros((512,512,3),np.uint8)
    cv2.namedWindow("paint")
    cv2.setMouseCallback("paint",onMouse,param=img)

    while True:
        cv2.imshow('paint', img)

        k=cv2.waitKey(1)&0xFF
        if k==27:
            break

    cv2.destroyAllWindows()

mouseBrush()