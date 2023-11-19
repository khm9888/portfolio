# cv2.createTrackbar(trackbarname, windowname, start, end, onChange): 트랙바를 지정된 윈도에 생성하는 함수
# trackbarname: 트랙바 이름
# windowname: 트랙바가 생성될 윈도 이름
# start: 트랙바 시작 값
# end: 트랙바 끝 값
# onChange: 트랙바 이벤트 발생시 수행되는 콜백 함수
# ​

# cv2.getTrackbarPos(trackbarname, windowname): 트랙바의 현재 위치를 리턴하는 함수
# trackbarname: 트랙바 이름
# windowname: 트랙바가 생성된 윈도 이름

import numpy as np
import cv2

def onChange(x):
    pass

def trackbar():
    img = np.zeros((200, 512, 3), np.uint8)
    cv2.namedWindow('color_palette')
    cv2.createTrackbar('B', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('G', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('R', 'color_palette', 0, 255, onChange)

    switch = '0 : OFF \n1: ON'
    cv2.createTrackbar(switch, 'color_palette', 0, 1, onChange)
    while True:
        cv2.imshow('color_palette', img)
        k=cv2.waitKey(1)&0xFF
        
        if k==27:
            break

        b = cv2.getTrackbarPos('B', 'color_palette')
        g = cv2.getTrackbarPos('G', 'color_palette')
        r = cv2.getTrackbarPos('R', 'color_palette')
        s = cv2.getTrackbarPos(switch, 'color_palette')

        if s == 0:
            img[:] = 0

        else:
            img[:] = [b, g, r]

trackbar()