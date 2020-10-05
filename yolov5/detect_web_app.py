import argparse
import os

from utils.datasets import *
from utils.utils import *

import warnings
warnings.filterwarnings('ignore')


def detect_web_app(dataset, 
                   conf_thres,
                   dataset_mode='image',
                   weights='./weights/best.pt',
                   half=False,
                   view_img=False,
                   iou_thres=0.5,
                   imgsz=640,
                   fourcc='mp4v',
                   device='') :

    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    # Initialize
    device = torch_utils.select_device(device)

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']

    # model.fuse()
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    data = letterbox(dataset, new_shape=imgsz)[0]
    
    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    prev_fname = ''
    # for img, vid_cap in dataset:
    img = data.copy()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.permute(2,0,1).unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()

    # to float
    if half:
        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    bndbxs = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(dataset.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], dataset.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                label = '%s %.2f' % (names[int(cls)], conf)
                bndbxs.append([xyxy, label, colors[int(cls)]])

    return bndbxs
