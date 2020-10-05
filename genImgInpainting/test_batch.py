import os
import random
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
import glob, time
import cv2
import pathlib
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from utils import datasets

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file
from utils.tools import default_loader, normalize, get_model_list, load_bbox_txt, create_mask
from ImageChunker import ImageChunker

import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--image', type=str)
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--overlap', type=int, default=0)

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    # print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    dataset = datasets.LoadImages(args.image)
    chunker = ImageChunker(config['image_shape'][0], 
                           config['image_shape'][1], 
                           args.overlap)
    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            # Set checkpoint path
            if not args.checkpoint_path:
                checkpoint_path = os.path.join('checkpoints', config['dataset_name'],
                                               config['mask_type'] + '_' + config['expname'])
            else:
                checkpoint_path = args.checkpoint_path
            last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)

            prev_fname = ''
            vid_writer = None
            for fpath, img_ori, vid_cap in dataset :
                imgs, masks = [], []
                if prev_fname == fpath :
                    frame += 1 # increase frame number if still on the same file
                else :
                    frame = 0 # start frame number
                    _, img_h, img_w = img_ori.shape
                    txtfile = pathlib.Path(fpath).with_suffix('.txt') # Load mask txt file
                    txtfile = os.path.join(args.output, str(txtfile).split('/')[-1])
                    if os.path.exists(txtfile) :
                        bboxes, bframes = load_bbox_txt(txtfile, img_w, img_h)
                    assert len(bboxes) == len(bframes)

                idx = [ii for ii, val in enumerate(bframes) if val==frame]
                bndbxs = [bboxes[ii] for ii in idx]
                img_ori = np.moveaxis(img_ori, 0, -1)
                if len(bndbxs) > 0 : # if any logo detected
                    mask_ori = create_mask(bndbxs, img_w, img_h)
                    # fig, axes = plt.subplots(1,2); axes[0].imshow(img_ori[0]); axes[1].imshow(mask_ori); plt.show()
                    chunked_images = chunker.dimension_preprocess(np.array(deepcopy(img_ori)))
                    chunked_masks = chunker.dimension_preprocess(np.array(deepcopy(mask_ori)))
                    for (x, msk) in zip(chunked_images, chunked_masks) :
                        x = transforms.ToTensor()(x)
                        mask = transforms.ToTensor()(msk)[0].unsqueeze(dim=0)
                        # x = normalize(x)
                        x = x * (1. - mask)
                        x = x.unsqueeze(dim=0)
                        mask = mask.unsqueeze(dim=0)
                        imgs.append(x)
                        masks.append(mask)

                    # Define the trainer
                    netG = Generator(config['netG'], cuda, device_ids)
                    netG.load_state_dict(torch.load(last_model_name))
                    model_iteration = int(last_model_name[-11:-3])
                    # print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

                    pred_imgs = []
                    for (x, mask) in zip(imgs, masks) :
                        if torch.max(mask) == 1 :
                            if cuda:
                                netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                                x = x.cuda()
                                mask = mask.cuda()

                            # Inference
                            x1, x2, offset_flow = netG(x, mask)
                            inpainted_result = x2 * mask + x * (1. - mask)
                            inpainted_result = inpainted_result.squeeze(dim=0).permute(1,2,0).cpu()
                            pred_imgs.append(inpainted_result.numpy())
                        else :
                            pred_imgs.append(x.squeeze(dim=0).permute(1,2,0).numpy())

                    pred_imgs = np.asarray(pred_imgs, dtype=np.float32)
                    reconstructed_image = chunker.dimension_postprocess(pred_imgs, np.array(img_ori))
                    reconstructed_image = np.uint8(reconstructed_image[:, :, ::-1]*255) # BGR to RGB, and rescaling
                else : # no logo detected
                    reconstructed_image = img_ori[:, :, ::-1]

                # Save results (image with detections)
                outname = fpath.split('/')[-1]
                outname = outname.split('.')[0] + '-inp.' + outname.split('.')[-1]
                outpath = os.path.join(args.output, outname)
                if dataset.mode == 'images':
                    cv2.imwrite(outpath, reconstructed_image)
                    print("Saved the inpainted image to {}".format(outpath))
                else :
                    if fpath != prev_fname:  # new video
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            print("Saved the inpainted video to {}".format(outpath))

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
                    vid_writer.write(reconstructed_image)
                    prev_fname = fpath                
    # exit no grad context
    except Exception as err:  # for unexpected error logging
        print("Error: {}".format(err))
        pass
    print('Inpainting: (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    main()
