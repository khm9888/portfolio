import numpy as np
import cv2

def writeVideo():
    try:
        print("run")
        cap = cv2.VideoCapture(0)
    except:
        print("fail")
        return

    fps =20.0
    width=int(cap.get(3))
    height=int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    out = cv2.VideoWriter('mycam.avi', fcc, fps, (width, height))
    print("녹화를 시작합니다.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("video read error")
            break
        cv2.imshow('video', frame)
        out.write(frame)
        
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

writeVideo()