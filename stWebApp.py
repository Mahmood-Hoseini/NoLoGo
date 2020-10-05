import streamlit as st
import numpy as np
import cv2
import os
import torch
import time

os.chdir('/home/idl/Documents/InsightProject/yolov5')
from detect_web_app import detect_web_app

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)


st.write('# DeepGreek: Logo Filtering')

# st.sidebar.text("image: [bmp, jpg, jpeg, png, tif, dng]")
uploaded_img = st.sidebar.file_uploader("Choose an image", 
                                    type=['bmp', 'jpg', 'jpeg', 'png', 'tif', 'dng'])

# st.sidebar.text("video: [mov, avi, mp4]")
uploaded_vid = st.sidebar.file_uploader("Choose a video", type=['mov', 'avi', 'mp4'])


@st.cache(allow_output_mutation=True)
def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    # Check if camera opened successfully
    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream


confidence_value = st.sidebar.slider('Confidence:', 0.0, 1.0, 0.2, 0.05)
if uploaded_img :
    st.sidebar.info('Uploaded image:')
    st.sidebar.image(uploaded_img, width=240)
    image = cv2.imdecode(np.fromstring(uploaded_img.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        bndbxs = detect_web_app(image, confidence_value, dataset_mode='image')

    for info in bndbxs :
        tl = 3  # line/font thickness
        color = info[2]
        c1, c2 = (int(info[0][0]), int(info[0][1])), (int(info[0][2]), int(info[0][3]))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(info[1], 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, info[1], (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], 
                        thickness=tf, lineType=cv2.LINE_AA)

    st.image(image, width=350)


    

if uploaded_vid :
    while True:
        image = get_cap(uploaded_vid).read()
        try:
            image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
        except:
            break

        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            bndbxs = detect_web_app(image, confidence_value, dataset_mode='image')

        for info in bndbxs :
            tl = 3  # line/font thickness
            color = info[2]
            c1, c2 = (int(info[0][0]), int(info[0][1])), (int(info[0][2]), int(info[0][3]))
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(info[1], 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, info[1], (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], 
                            thickness=tf, lineType=cv2.LINE_AA)

        st.image(image, width=350)
