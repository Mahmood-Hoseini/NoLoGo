import streamlit as st
import numpy as np
import cv2
import os
import torch
import time
import io
import glob
import shutil
from streamlit import caching

# os.chdir('/home/idl/Documents/NoLoGo')

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=False)
def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    # Check if camera opened successfully
    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream


st.write('# NoLoGo: Logo Filtering')

caching.clear_cache()
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
mode = st.sidebar.radio("input type: ", ("image", "video"))

confidence_value = st.sidebar.slider('Confidence:', 0.0, 1.0, 0.2, 0.05)
if mode == "image" :
    up_img = st.sidebar.file_uploader("Choose an image", 
                            type=['bmp', 'jpg', 'jpeg', 'png', 'tif', 'dng'])

    if up_img :
        st.sidebar.info('Uploaded image:')
        st.sidebar.image(up_img, width=240)
        g = io.BytesIO(up_img.read())  ## BytesIO Object

        temp_loc = "./data/tmp/"
        out_loc = "./data/tmp_output/"
        if os.path.exists(temp_loc) : shutil.rmtree(temp_loc)
        if os.path.exists(out_loc) : shutil.rmtree(out_loc)
        os.makedirs(temp_loc)
        os.makedirs(out_loc)

        with open(temp_loc+"testout_img.jpg", 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        os.chdir('./yolov5/')
        yolo = f"python detect.py --source ../data/tmp/ --weights ./weights/best.pt --output ../data/tmp_output --img-size 512 --conf-thres {confidence_value} --save-txt"
        os.system(yolo)
        st.sidebar.success("Detection Done!")

        os.chdir('../genImgInpainting/')
        imgInp = f"python test_batch.py --image ../data/tmp --output ../data/tmp_output --config configs/config.yaml"
        os.system(imgInp)
        st.sidebar.success("Inpainting Done!")

        os.chdir('../')
        c1, c2 = st.beta_columns(2)
        det_file = glob.glob(out_loc + '*-det.jpg')[0]
        det_img = cv2.imread(det_file)
        image1 = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        st.image(image1, caption='logos detected', width=500)

        inp_file = glob.glob(out_loc + '*-inp.jpg')[0]
        inp_img = cv2.imread(inp_file)
        image2 = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        st.image(image2, caption='inpainted', width=500)
    
elif mode == "video" :
    up_vid = st.sidebar.file_uploader("Choose a video", type=['mov', 'avi', 'mp4'])
    if up_vid :
        g = io.BytesIO(up_vid.read())  ## BytesIO Object

        temp_loc = "./data/tmp/"
        out_loc = "./data/tmp_output/"
        if os.path.exists(temp_loc) : shutil.rmtree(temp_loc)
        if os.path.exists(out_loc) : shutil.rmtree(out_loc)
        os.makedirs(temp_loc)
        os.makedirs(out_loc)

        with open(temp_loc+"testout_vid.webm", 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        # video2 = open("/home/idl/Documents/DeepGreek/data/OUT.webm", 'rb').read()
        # st.video(video2)

        os.chdir('./yolov5/')
        yolo = f"python detect.py --source ../data/tmp/ --weights ./weights/best.pt --output ../data/tmp_output --img-size 512 --conf-thres {confidence_value} --save-txt"
        print(yolo)
        os.system(yolo)
        st.sidebar.success("Detection Done!")

        os.chdir('../genImgInpainting/')
        imgInp = f"python test_batch.py --image ../data/tmp --output ../data/tmp_output --config configs/config.yaml"
        os.system(imgInp)
        st.sidebar.success("Inpainting Done!")

        os.chdir('../')
        c1, c2 = st.beta_columns(2)
        out_loc = './data/tmp_output/'
        det_file = glob.glob(out_loc + '*-det.webm')[0]
        # web_det_file = det_file[:-3] + "webm"
        # os.system(f"ffmpeg -i {det_file} -f webm -vcodec libvpx -ab 128000 {web_det_file}")
        video1 = open(det_file, 'rb').read()
        c1.video(video1)

        inp_file = glob.glob(out_loc + '*-inp.webm')[0]
        # web_inp_file = inp_file[:-3] + "webm"
        # os.system(f"ffmpeg -i {inp_file} -f webm -vcodec libvpx -ab 128000 {web_inp_file}")
        video2 = open(inp_file, 'rb').read()
        c2.video(video2)