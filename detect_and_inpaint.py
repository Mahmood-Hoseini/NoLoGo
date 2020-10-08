import argparse
import os
import time
import torch

def detect_and_inpaint(opt):
    print(opt)

    t0 = time.time()
    os.chdir('./yolov5/')
    yolo = f"python detect.py --source {opt.images} --weights {opt.yoloWeights} --output {opt.output} --img-size {opt.yoloImgSize} --conf-thres {opt.conf} --iou-thres {opt.iou} --fourcc {opt.fourcc} --save-txt"
    os.system(yolo)
    
    os.chdir('../genImgInpainting/')
    imgInp = f"python test_batch.py --image {opt.images} --output {opt.output} --config {opt.config} --iter {opt.iter} --overlap {opt.overlap}"
    print(imgInp)
    os.system(imgInp)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloWeights', type=str, default='./weights/best.pt', help='model.pt path')
    parser.add_argument('--images', type=str, default='../data/images', help='source images')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='../data/outputs', help='output folder')  # output folder
    parser.add_argument('--yoloImgSize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='VP90', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help="training configuration")
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--overlap', type=int, default=0)

    opt = parser.parse_args()

    with torch.no_grad():
        detect_and_inpaint(opt)
