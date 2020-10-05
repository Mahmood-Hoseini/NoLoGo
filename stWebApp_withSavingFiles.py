import streamlit as st
import numpy as np
import cv2
import os
import torch
import time
import io
import glob

os.chdir('/home/idl/Documents/InsightProject')

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    # Check if camera opened successfully
    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream


st.write('# DeepGreek: Logo Filtering')

st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
mode = st.sidebar.radio("input type: ", ("image", "video"))

confidence_value = st.sidebar.slider('Confidence:', 0.0, 1.0, 0.2, 0.05)
if mode == "image" :
    # st.sidebar.text(".bmp, .jpg, .jpeg, .png, .tif")
    up_file = st.sidebar.file_uploader("Choose an image", 
                            type=['bmp', 'jpg', 'jpeg', 'png', 'tif', 'dng'])
    st.sidebar.info('Uploaded image:')
    st.sidebar.image(up_file, width=240)
    g = io.BytesIO(up_file.read())  ## BytesIO Object
    temp_loc = "./data/tmp/testout_img.jpg"
    with open(temp_loc, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file

    os.chdir('./yolov5/')
    # yolo = f"python detect.py --source ../data/tmp/ --weights ./weights/best.pt --output ../data/tmp_output --img-size 512 --conf-thres {confidence_value} --save-txt"
    # os.system(yolo)

    os.chdir('../genImgInpainting/')
    # imgInp = f"python test_batch.py --image ../data/tmp --output ../data/tmp_output --config configs/config.yaml"
    # os.system(imgInp)

    os.chdir('../')
    c1, c2 = st.beta_columns(2)
    temp_loc = './data/tmp_output/'
    det_file = glob.glob(temp_loc + '*-det.jpg')[0]
    det_img = cv2.imread(det_file)
    image1 = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    c1.image(image1, caption='logo detection')

    inp_file = glob.glob(temp_loc + '*-inp.jpg')[0]
    inp_img = cv2.imread(inp_file)
    image2 = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
    c2.image(image2, caption='inpainting')
    
elif mode == "video" :
    up_file = st.sidebar.file_uploader("Choose a video", type=['mov', 'avi', 'mp4'])
    g = io.BytesIO(up_file.read())  ## BytesIO Object
    temp_loc = "./data/tmp/testout_vid.mp4"

    with open(temp_loc, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file

    os.chdir('./yolov5/')
    # yolo = f"python detect.py --source ../data/tmp/ --weights ./weights/best.pt --output ../data/tmp_output --img-size 512 --conf-thres {confidence_value} --save-txt"
    # os.system(yolo)

    os.chdir('../genImgInpainting/')
    # imgInp = f"python test_batch.py --image ../data/tmp --output ../data/tmp_output --config configs/config.yaml"
    # os.system(imgInp)

    os.chdir('../')
    c1, c2 = st.beta_columns(2)
    temp_loc = './data/tmp_output/'
    det_file = glob.glob(temp_loc + '*-det.mp4')[0]
    video1 = open(det_file, 'rb').read()
    c1.video(video1)

    inp_file = glob.glob(temp_loc + '*-inp.mp4')[0]
    video2 = open(inp_file, 'rb').read()
    c2.video(video2)


#     scaling_factorx = 0.25
#     scaling_factory = 0.25
#     image_placeholder = st.empty()
#     temp_loc = './data/tmp_output/testout_vid-det.mp4'
#     while True:
#         # here it is a CV2 object
#         video_stream = get_cap(temp_loc)
#         # video_stream = video_stream.read()
#         ret, image = video_stream.read()
#         if ret:
#             image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
#         else:
#             print("there was a problem or video was finished")
#             video_stream.release()
#             break
#         # check if frame is None
#         if image is None:
#             print("there was a problem None")
#             # if True break the infinite loop
#             break

#         image_placeholder.image(image, channels="RGB")

# #     video_stream.release()




