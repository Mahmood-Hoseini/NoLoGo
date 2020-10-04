# DeepGreek: smart logo replacement using image inpainting

## Motivation
Greeking out logos and trademarks are is the of physically obscuring company logos. That’s a reference to the phrase “Greek to me,” because the final version is unrecognizable. Having company logos in an image/video is not always desirable especially for celebrities as well as film/TV show production companies. Social media bloggers and celebrities do not want to give away free ads and therefore are always concerned what logos are visible in their social media posts. While showing logos and labels in a TV show is not illegal, production companies are going to err on the side of extreme caution. On reality TV, where budgets are tight, the greeking process often occurs on the scene during filming, with tape or markers which is often comical and makes logos/labels more noticeable. Ted Allen revealed in an interview, “We have a graphic designer who sits there right next to the Chopped kitchen and prints out with this elaborate printer all of these crazy labels that she’s designed." Hereby, I created a deep learning pipeline that can automatically detect logos/labels from images/videos and replace them with the context of the image/video.

Google slides for the project can be found [here]().

## Overview of the project:
Deep greeking is implemented in two steps:

- Logo Detection using YOLOv5 pytorch model trained on [OpenLogo](https://qmul-openlogo.github.io/) dataset
- Inpaint logo area using [generative image inpainting with contextual attention](https://arxiv.org/abs/1801.07892)

## Prerequisites
This code has been tested on Ubuntu 18.04 and the following are the main components that need to be installed:
- Python3
- matplotlib 
- numpy 1.17.0
- scipy 1.4.1
- tensorboard 2.2.1
- torch 1.5.0
- torchvision 0.6.0

## Logo detection using YOLOv5
**Preparing Dataset**: Once you get the labeled OpenLogo dataset, generate annotations in YOLO format (see this [notebook]()), divide it into 80% training set, 10% validation set, and 10 % in the testing set, and modify `./yolov5/data/LOGO.yaml` file by adding appropriate data paths

**Training**: Here, I'm using small version of YOLO. To train it just run:

```bash
cd DeepGreek/yolov5/
python train.py --img 512 --batch 16 --epochs 100 --data ./data/LOGO.yaml --cfg ./models/yolov5s.yaml --weights '' --device 0
```

**Testing**: Now that the model has been trained, you can test its performance:

```bash
python detect.py --source ./data/test-logo  --weights ./weights/best.pt --conf 0.3 --save-txt
```

## Generative image inpainting
A PyTorch reimplementation for the paper [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892) according to the author's [TensorFlow implementation](https://github.com/JiahuiYu/generative_inpainting). Also see this [github repository](https://github.com/daa233/generative-inpainting-pytorch).

**Train the model**: To train the inpainting model, first modify config.yaml file. To be able to process any image/video size while keeping the training doable, set `image_shape: [256, 256, 3]` and chunk any input image into appropriate size (see this [notebook]() and [Image_chunker.py]()). To train run:
```bash

cd DeepGreek/genImgInpainting/
python train.py --config ./configs/config.yaml
```
The checkpoints and logs will be saved to `checkpoints`.

## Test with the trained model
By default, it will load the latest saved model in the checkpoints. You can also use `--iter` to choose the saved models by iteration.

Trained PyTorch model: [[Google Drive](https://drive.google.com/open?id=1qbfA5BP9yzdTFFmiOTvYARUYgW1zwBBK)] [[Baidu Wangpan](https://pan.baidu.com/s/17HzpiqMPLIznvCWBfpNVGw)]

```bash
python test_single.py \
	--image examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png \
	--mask examples/center_mask_256.png \
	--output examples/output.png
```

## Test with the converted TF model:
Converted TF model: [[Google Drive](https://drive.google.com/file/d/1vz2Qp12_iwOiuvLWspLHrC1UIuhSLojx/view?usp=sharing)]

```bash
python test_tf_model.py \
	--image examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png \
	--mask examples/center_mask_256.png \
	--output examples/output.png \
	--model-path torch_model.p
```

## Test results on ImageNet validation set patches

With PyTorch, the model was trained on ImageNet for 430k iterations to converge (with batch_size 48, about 150h). Here are some test results on the patches from ImageNet validation set.

| Input | Inpainted |
|:---:|:---:|
| [![val_00000827_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_input.png)  | [![val_00000827_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_output.png) |
| [![val_00008210_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png)  | [![val_00008210_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_output.png) |
| [![val_00022355_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_input.png)  | [![val_00022355_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_output.png) |
| [![val_00025892_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_input.png)  | [![val_00025892_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_output.png) |
| [![val_00045643_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_input.png)  | [![val_00045643_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_output.png) |

## References:

1. Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, Thomas Huang. 2018. Generative Image Inpainting with Contextual Attention. [link](https://arxiv.org/abs/1801.07892).

2. Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro. 2018. Image Inpainting for Irregular Holes Using Partial Convolutions. [link](https://arxiv.org/abs/1804.07723).

3. New AI Imaging Technique Reconstructs Photos with Realistic Results [link](https://news.developer.nvidia.com/new-ai-imaging-technique-reconstructs-photos-with-realistic-results/?ncid=nv-twi-37107).
