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
import base64

# os.chdir('/home/idl/Documents/NoLoGo')

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
    max-width: 1000px;
    padding-top: 2.0rem;
    padding-right: 2.0rem;
    padding-left: 2.0rem;
    padding-bottom: 2.0rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

set_png_as_page_bg("./background.png")
logo = open("./nologo.png", 'rb').read()
st.image(logo, width=200)
st.write('## Smart Logo Replacement Using Image Inpainting')

st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
mode = st.sidebar.radio("input type: ", ("image", "video"))
caching.clear_cache()

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
        c1.image(image1, caption='logos detected', width=400)

        inp_file = glob.glob(out_loc + '*-inp.jpg')[0]
        inp_img = cv2.imread(inp_file)
        image2 = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        c2.image(image2, caption='inpainted', width=400)
    
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

        with open(temp_loc+"testout_vid.mp4", 'wb') as out:  ## Open temporary file as bytes
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
        det_file = glob.glob(out_loc + '*-det.mp4')[0]
        os.system(f"ffmpeg -i {det_file} -vcodec libx264 {det_file.replace('-det.', '-det2.')}")
        video1 = open(det_file.replace('-det.', '-det2.'), 'rb').read()
        c1.video(video1)

        inp_file = glob.glob(out_loc + '*-inp.mp4')[0]
        os.system(f"ffmpeg -i {inp_file} -vcodec libx264 {inp_file.replace('-inp.', '-inp2.')}")
        video2 = open(inp_file.replace('-inp.', '-inp2.'), 'rb').read()
        c2.video(video2)