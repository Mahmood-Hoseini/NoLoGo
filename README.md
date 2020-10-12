[![build status](https://img.shields.io/badge/build-passing-green.svg)]()

[![build status](https://img.shields.io/badge/made%20with-python-cyan.svg)](https://www.python.org/)
[![version](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)
[![build status](https://img.shields.io/badge/opencv-v4.2.0.34-gold.svg)](https://pypi.org/project/opencv-python/)

# NoLogo: smart logo replacement using image inpainting

## Motivation
Having company logos in an image/video is not always desirable especially for celebrities as well as film/TV show production companies. Social media bloggers and celebrities do not want to give away free ads and therefore are always concerned what logos are visible in their social media posts. While showing logos and labels in a TV show is not illegal, production companies are going to err on the side of extreme caution. On reality TV, where budgets are tight, the obscuring process often occurs on the scene during filming, with tape or markers which is often comical and makes logos/labels more noticeable. Ted Allen revealed in an interview, “We have a graphic designer who sits there right next to the Chopped kitchen and prints out with this elaborate printer all of these crazy labels that she’s designed." Hereby, I created a deep learning pipeline that can automatically detect logos/labels from images/videos and replace them with the context of the image/video.

Google slides for the project can be found [here](https://docs.google.com/presentation/d/1zeWUrgL25nQvZic-l-hdXLsXnGBdtkK04DWaZ5r5QCs/edit?usp=sharing).

A simple [Streamlit frontend](http://54.67.95.39:8501/) was designed that receives images/videos and returns inpainted output.
![]()

## Overview of the project:
NoLoGo is implemented in two steps:

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
- PyYAML 5.3.1
- opencv
- streamlit

## Logo detection using YOLOv5
**Preparing Dataset**: Once you get the labeled OpenLogo dataset, generate annotations in YOLO format (see this [notebook](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/notebooks/Step1%20-%20Processing%20openlogo%20dataset.ipynb)), divide it into 80% training set, 10% validation set, and 10 % in the testing set, and modify `./yolov5/data/LOGO.yaml` file by adding appropriate data paths

**Training**: Here, I'm using small version of YOLO. To train it just run:
```bash
cd NoLoGo/yolov5/
python train.py --img 512 --batch 16 --epochs 100 --data ./data/LOGO.yaml --cfg ./models/yolov5s.yaml --weights '' --device 0
```

**Testing**: Now that the model has been trained, you can test its performance:
```bash
python detect.py --source ./data/test-logo  --weights ./weights/best.pt --conf 0.3 --save-txt
```

## Generative image inpainting
A PyTorch reimplementation for the paper [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892) according to the author's [TensorFlow implementation](https://github.com/JiahuiYu/generative_inpainting). Also see this [github repository](https://github.com/daa233/generative-inpainting-pytorch).

**Training**: To train the inpainting model, first modify config.yaml file. To be able to process any image/video size while keeping the training doable, set `image_shape: [256, 256, 3]` and chunk any input image into appropriate size (see this [notebook](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/notebooks/Step3%20-%20Prediction.ipynb) and [Image_chunker.py](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/genImgInpainting/ImageChunker.py)). To train run:
```bash
cd NoLoGo/genImgInpainting/
python train.py --config ./configs/config.yaml
```
With PyTorch, the model was trained on the dataset 500,000 iterations (with batch_size 64, ~80 hrs). The checkpoints and logs will be saved to `cpts`.

**Testing**: Test the latest saved model on a single image/video by runing:
```bash
cd NoLoGo/genImgInpainting/
python test_single.py --image ../data/sample_img.jpg --mask ../data/sample_mask.jpg\
                      --output ../data/output_img.jpg  --config ./configs/config.yaml
```
To test the latest saved model in batch run:
```bash
cd NoLoGo/genImgInpainting/
python test_batch.py --image ../data/ --output ../data/outputs/ --config ./configs/config.yaml
```

## Detect and inpaint
Now we can put everything together by running:
```bash
cd ./NoLoGo/
python detect_and_inpaint.py --
```
To run th streamlit webapp execute `streamlit run stWebApp_withSavingFiles.py`.

## Test results
Here are some test results

| Input | Detected | Inpainted |
|:---:|:---:|:---:|
| ![img1](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/2105646918.jpg)  | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/2105646918-det.jpg) | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/2105646918-inp.jpg) |
| ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/2659660776.jpg)  | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/2659660776-det.jpg) | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/2659660776-inp.jpg) |
| ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/5077581837.jpg)  | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/5077581837-det.jpg) | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/5077581837-inp.jpg) |
| ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/898312343.jpg)  | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/898312343-det.jpg) | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/898312343-inp.jpg) |
| ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/logos32plus_000573.jpg)  | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/logos32plus_000573-det.jpg) | ![](https://github.com/Mahmood-Hoseini/NoLoGo/blob/master/data/outputs/logos32plus_000573-inp.jpg) |

## References:
1. Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, Thomas Huang. 2018. Generative Image Inpainting with Contextual Attention. [link](https://arxiv.org/abs/1801.07892).
2. Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro. 2018. Image Inpainting for Irregular Holes Using Partial Convolutions. [link](https://arxiv.org/abs/1804.07723).
3. New AI Imaging Technique Reconstructs Photos with Realistic Results [link](https://news.developer.nvidia.com/new-ai-imaging-technique-reconstructs-photos-with-realistic-results/?ncid=nv-twi-37107).
