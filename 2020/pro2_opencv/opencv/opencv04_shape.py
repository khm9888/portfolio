# cv2.line(): 직선 그리기 함수
# cv2.circle(): 원 그리기 함수
# cv2.rectangle(): 직사각형 그리기 함수
# cv2.ellipse(): 타원 그리기 함수
# cv2.putText(): 텍스트 입력 함수


# 위에서 나열한 함수에 공통적으로 적용되는 인자가 있는데, 그 인자들은 다음과 같습니다. 인자 이름은 임의로 제가 작명한 것이고, 프로그래밍할 때 여러분이 원하는 인자 이름으로 바꾸어 적용하면 됩니다.



# img: 각종 도형을 그리기 위한 공간. 결국은 img를 화면에 디스플레이 하게 됨
# color: 도형 색상. OpenCV는 BGR 모드를 이용하므로, (255, 0, 0)은 청색, (0, 255, 0)은 녹색, (0, 0, 255)는 빨간색을 나타냄
# thickness: 선 굵기. 원, 직사각형과 같이 닫힌 도형에서 -1 값을 전달하면 도형을 채우게 됨. 디폴트 값은 1.
# lineType: 선을 화면에 표현하는 방법. 8-connected 또는 anti-aliased line 등으로 설정 가능. 디폴트는 8-connected 임. cv2.LINE_AA로 설정하면 곡선에서 유용한 anti-aliased line으로 화면에 선을 그림

import numpy as np
import cv2

def drawing():
    img = np.zeros((512,512,3),np.uint8)

    cv2.line(img, (0,0),(511,511),(255,0,0),5)
    cv2.rectangle(img, (384, 0), (510,128), (0,255,0), 3)
    cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    cv2.ellipse(img, (256,256), (100,50), 0, 0, 180, (255, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "OpenCV",(10,500),font,4,(255,255,255),2)

    cv2.imshow("drawing",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

drawing()
