import streamlit as st
import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import pixellib
from pixellib.semantic import semantic_segmentation
indx = {'Basketball':0, 'Biking':1, 'GolfSwing':2, 'YoYo':3}
model = tf.keras.models.load_model('VidClass.h5')
def run_on_video(output_file_path):
    st.title("Video  Classification")
    uploaded_video = st.file_uploader("Choose video for classification", type=["mp4",'avi'])
    
    if uploaded_video != None :
    # Reading the Video File using the VideoCapture Object
        cap = cv2.VideoCapture(uploaded_video.name)

    # Getting the width and height of the video 
    original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'x264'), 24, (original_video_width, original_video_height))
    
    cap.set(cv2.CAP_PROP_FPS, 1)
    frameRate=cap.get(5)
    x=1
    count=0
    vid_data = []
    copy = (64, 64, 1)
    while(cap.isOpened()) :
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            frame_grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./frame.jpg', frame_grey)
            image=Image.open('./frame.jpg')
            image=image.resize((64, 64), Image.ANTIALIAS) 
            datu=np.asarray(image)
            normu_dat=datu/255
            vid_data.append(normu_dat)
            count += 1
    cap.release()
    if count < 8 :
        marge = 8 - count 
        for i in range(marge) :
            vid_data.append(np.zeros(shape=copy))
    elif count > 8 :
        vid_data = vid_data[:8]
    ppred = model.predict(np.expand_dims(vid_data, axis = 0))[0]
    pred = np.argmax(ppred)
    label = list(indx.keys())[pred]
    
    st.text("The video is about : {}".format(label))
    cap.release()
    video_writer.release()
    
    ########################################################################################
    
    st.title("Video  Segmentation")
    uploaded_video = st.file_uploader("Choose video for segmentation", type=["mp4",'avi'])
    
    if uploaded_video != None :
    # Reading the Video File using the VideoCapture Object
    name = uploaded_video.name
        cap = cv2.VideoCapture(name)

    # Getting the width and height of the video 
    original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'x264'), 24, (original_video_width, original_video_height))
    
    # Creating a semantic_segmentation object
    segment_video = semantic_segmentation()
    # Loading the Xception model trained on ade20k dataset
    segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
    # Processing the video
    segment_video.process_video_ade20k(
        name, 
        frames_per_second=1,
        overlay = True,
        output_video_name="semantic_seg_output.mp4",
    )
    cap.release()
    video_writer.release()
    
    
def main() :
    run_on_video('output.mp4')

if __name__ == "__main__":
    main()
