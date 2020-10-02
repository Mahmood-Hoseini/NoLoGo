import os
import random
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

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
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)


    chunker = ImageChunker(config['image_shape'][0], config['image_shape'][1], args.overlap)
    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            if is_image_file(args.image):
                print("Loading image...")
                imgs, masks = [], []
                img_ori = default_loader(args.image)
                img_w, img_h = img_ori.size
                # Load mask txt file
                fname = args.image.replace('.jpg', '.txt')
                bboxes, _ = load_bbox_txt(fname, img_w, img_h)
                mask_ori = create_mask(bboxes, img_w, img_h)
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

                # Set checkpoint path
                if not args.checkpoint_path:
                    checkpoint_path = os.path.join('checkpoints', config['dataset_name'],
                                                   config['mask_type'] + '_' + config['expname'])
                else:
                    checkpoint_path = args.checkpoint_path

                # Define the trainer
                netG = Generator(config['netG'], cuda, device_ids)
                # Resume weight
                last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
                netG.load_state_dict(torch.load(last_model_name))
                model_iteration = int(last_model_name[-11:-3])
                print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

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
                # plt.imshow(reconstructed_image); plt.show()
                reconstructed_image = torch.tensor(reconstructed_image).permute(2,0,1).unsqueeze(dim=0)
                vutils.save_image(reconstructed_image, args.output, padding=0, normalize=True)
                print("Saved the inpainted result to {}".format(args.output))
                if args.flow:
                    vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
                    print("Saved offset flow to {}".format(args.flow))
            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
