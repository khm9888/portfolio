
import sys
import os
# sys.path.append("D:\programming\mmdetection")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from mmdetection.mmdet.apis import init_detector, inference_detector
# from mmdet.apis import init_detector, inference_detector
import mmcv


mmdet_path = "D:\programming\mmdetection/"

config_file = f'{mmdet_path}configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = f'{mmdet_path}checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

img=f'{mmdet_path}demo/demo.jpg'
result=inference_detector(model, img)

model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file=f'{mmdet_path}demo/demo_result.jpg')

# test a video and show the results
video = mmcv.VideoReader(f'{mmdet_path}demo/demo.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)