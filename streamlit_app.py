import os
os.environ["MY_ENV_VARIABLE"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np

import cv2
import json
import subprocess

from deep_sort_realtime.deepsort_tracker import DeepSort
from _collections import deque
from tqdm import tqdm 
from collections import Counter
import time
import shutil

from ultralytics import YOLO
from ultralytics.engine.results import Results
from model_utils import get_yolo, get_system_stat
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, webrtc_streamer
from DistanceEstimation import *
from streamlit_autorefresh import st_autorefresh

import av
import torch
import intel_extension_for_pytorch as ipex
from pathlib import Path
import openvino as ov
import streamlit as st


# colors for visualization for image visualization
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
          (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]



def result_to_json(result: Results, tracker=None):
    """
    Convert result from ultralytics YOLOv8 prediction to json format
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        tracker: DeepSort tracker
    Returns:
        result_list_json: detection result in json format
    """
    len_results = len(result.boxes)
    result_list_json = [
        {
            'class_id': int(result.boxes.cls[idx]),
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.xyxy[idx][0]),
                'y_min': int(result.boxes.xyxy[idx][1]),
                'x_max': int(result.boxes.xyxy[idx][2]),
                'y_max': int(result.boxes.xyxy[idx][3]),
            },
        } for idx in range(len_results)
    ]
   
    if result.masks is not None:
       for idx in range(len_results):
           result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx].cpu().numpy(),(result.orig_shape[1], result.orig_shape[0])).tolist()  
           result_list_json[idx]['segments'] = [seg.tolist() for seg in result.masks.xy[idx]]
              
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_min'],
                    result_list_json[idx]['bbox']['x_max'] - result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_max'] - result_list_json[idx]['bbox']['y_min']
                ],
                result_list_json[idx]['confidence'],
                result_list_json[idx]['class'],
            ) for idx in range(len_results)
        ]
        tracks = tracker.update_tracks(bbs, frame=result.orig_img)
        for idx in range(len(result_list_json)):
            track_idx = next((i for i, track in enumerate(tracks) if track.det_conf is not None and np.isclose(track.det_conf, result_list_json[idx]['confidence'])), -1)
            if track_idx != -1:
                result_list_json[idx]['object_id'] = int(tracks[track_idx].track_id)
    return result_list_json



def view_result_ultralytics(result: Results, result_list_json, centers=None):
    """
    Visualize result from ultralytics YOLOv8 prediction using ultralytics YOLOv8 built-in visualization function
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        result_list_json: detection result in json format
        centers: list of deque of center points of bounding boxes
    Returns:
        result_image_ultralytics: result image from ultralytics YOLOv8 built-in visualization function
    """
   
    result_image_ultralytics = result.plot()
    for result_json in result_list_json:
        class_color = COLORS[result_json['class_id'] % len(COLORS)]
        if 'object_id' in result_json and centers is not None:
            centers[result_json['object_id']].append((int((result_json['bbox']['x_min'] + result_json['bbox']['x_max']) / 2), int((result_json['bbox']['y_min'] + result_json['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[result_json['object_id']])):
                if centers[result_json['object_id']][j - 1] is None or centers[result_json['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(result_image_ultralytics, centers[result_json['object_id']][j - 1], centers[result_json['object_id']][j], class_color, thickness)
    return result_image_ultralytics


def view_result_default(result: Results, result_list_json, centers=None, image=None):
    """
    Visualize result from ultralytics YOLOv8 prediction using default visualization function
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        result_list_json: detection result in json format
        centers: list of deque of center points of bounding boxes
    Returns:
        result_image_default: result image from default visualization function
    """
          
    ALPHA = 0.5
    # image = result.orig_img
    result_image_ultralytics = image.copy() if image is not None else result.orig_img.copy()
          
    for result in result_list_json:
        class_color = COLORS[result['class_id'] % len(COLORS)]
        fontScale = 1
        if 'mask' in result:
            image_mask = np.stack([np.array(result['mask']) * class_color[0], np.array(result['mask']) * class_color[1], np.array(result['mask']) * class_color[2]], axis=-1).astype(np.uint8)
            image = cv2.addWeighted(image, 1, image_mask, ALPHA, 0)
        text = f"{result['class']} {result['object_id']}: {result['confidence']:.2f}" if 'object_id' in result else f"{result['class']}: {result['confidence']:.2f}"
        cv2.rectangle(image, (result['bbox']['x_min'], result['bbox']['y_min']), (result['bbox']['x_max'], result['bbox']['y_max']), class_color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.90, 5)
        cv2.rectangle(image, (result['bbox']['x_min'], result['bbox']['y_min'] - text_height - baseline), (result['bbox']['x_min'] + text_width, result['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text , (result['bbox']['x_min'], result['bbox']['y_min'] - baseline), cv2.FONT_HERSHEY_DUPLEX, 0.90, (255, 255, 255), 1)
        if 'object_id' in result and centers is not None:
            centers[result['object_id']].append((int((result['bbox']['x_min'] + result['bbox']['x_max']) / 2), int((result['bbox']['y_min'] + result['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[result['object_id']])):
                if centers[result['object_id']][j - 1] is None or centers[result['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image, centers[result['object_id']][j - 1], centers[result['object_id']][j], class_color, thickness)
    return image


def image_processing(frame, model, image_viewer=view_result_default, tracker=None, centers=None):
    """
    Process image frame using ultralytics YOLOv8 model and possibly DeepSort tracker if it is provided
    Parameters:
        frame: image frame
        model: ultralytics YOLOv8 model
        image_viewer: function to visualize result, default is view_result_default, can be view_result_ultralytics
        tracker: DeepSort tracker
        centers: list of deque of center points of bounding boxes
    Returns:
        result_image: result image with bounding boxes, class names, confidence scores, object masks, and possibly object IDs
        result_list_json: detection result in json format
    """    
    # Preserve the original image for visualization
    original_image = frame.copy()
          
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (1, 1), 0)
    # equalized_image = cv2.equalizeHist(denoised_image)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)
    image = cv2.merge([enhanced_image, enhanced_image, enhanced_image])    
    processed_image = image
          
    st.image(processed_image, caption="Processed image", channels="BGR")
    results = model.predict(processed_image)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image = image_viewer(results[0], result_list_json, centers=centers, image=original_image)
    return result_image, result_list_json


def video_processing(video_file, model, image_viewer=view_result_default, tracker=None, centers=None):
    results = model.predict(video_file)
    model_name = model.ckpt_path.split('/')[-1].split('.')[0]

    output_folder = os.path.join('output_videos', os.path.splitext(os.path.basename(video_file))[0])
    os.makedirs(output_folder, exist_ok=True)
    video_file_name_out = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_file))[0]}_{model_name}_output.mp4")
    result_video_json_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_file))[0]}_{model_name}_output.json")
    
    for file_path in [video_file_name_out, result_video_json_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    json_file = open(result_video_json_file, 'w')
    first_frame = results[0].orig_img
    height, width = first_frame.shape[:2]
    video_writer = cv2.VideoWriter(video_file_name_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    result_list = []
    frame_count = 0

    for result in tqdm(results, desc="Processing video"):
        result_list_json = result_to_json(result, tracker=tracker)
        result_image = image_viewer(result, result_list_json, centers=centers)
        
        video_writer.write(result_image)
        result_list.append(result_list_json)
        frame_count += 1

    json.dump(result_list, json_file, indent=2)
    json_file.close()

    video_writer.release()

    if frame_count == 0 or os.path.getsize(video_file_name_out) == 0:
        raise FileNotFoundError(f"The video file {video_file_name_out} was not created or is empty.")

    return video_file_name_out, result_video_json_file
          
st.image("assets/nsidelogoo.png")
st.sidebar.image("assets/nsidelogoo.png")


def load_vino_model(model_path, device="CPU"):
    core = ov.Core()
    ov_model = core.read_model(model_path)
    ov_config = {}
    if device != "CPU":
        ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

    compiled_model = core.compile_model(ov_model, device, ov_config)

    return compiled_model


def infer_with_vino_model(compiled_model, *args):
    result = compiled_model(args)
    return torch.from_numpy(result[0])

@st.cache_resource
def load_model(model_path):
    # Load and return the YOLO model
    return YOLO(model_path)

device = "CPU"

model_vino_path = Path("yolov8xcdark_openvino_model/")
model_openvino = load_model(model_vino_path)

# model_openvino = load_vino_model(model_vino_path, device)


model_path = "yolov8xcdark.pt" 
model = load_model(model_path)

          
st.write("Optimized Openvino Yolov8c Models loaded successfully!")
model_seg_path = "yolov8xcdark-seg.pt"
model_seg = load_model(model_seg_path)


source = ("Image Detection📸", "Video Detections📽️", "Live Camera Detection🤳🏻","RTSP","MOBILE CAM")
source_index = st.sidebar.selectbox("Select Input type", range(
        len(source)), format_func=lambda x: source[x])


# Image detection section
if source_index == 0:
    st.title("Image Processing using YOLOv8c Dark Detector")
    image_file = st.file_uploader("Upload an image 🔽", type=["jpg", "jpeg", "png"])
    process_image_button = st.button("Detect")
    process_seg_button = st.button("Click here for Segmentation result")

    with st.spinner("Detecting with 💕"):
        if image_file is None and process_image_button:
            st.warning("Please upload an image file to be processed!")
        if image_file is not None and process_image_button:
            st.write(" ")
            st.sidebar.success("Successfully uploaded")
            st.sidebar.image(image_file, caption="Uploaded image")
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            
            img, result_list_json = image_processing(img, model_openvino)
          
            st.success("✅ Task Detect : Detection using custom-trained v8 model")
            st.image(img, caption="Detected image", channels="BGR")     
            
            detected_classes = [item['class'] for item in result_list_json]
            class_fq = Counter(detected_classes)
            
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
          
            st.write("Class Frequency:")
            st.dataframe(df_fq)  
            
        if image_file is not None and process_seg_button:
            st.write(" ")
            st.sidebar.success("Successfully uploaded")
            st.sidebar.image(image_file,caption="Uploaded image")
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1) 
           
            ## for detection with bb & segmentation masks
            img, result_list_json = image_processing(img, model_seg)
            st.success("✅ Task Segment: Segmentation using v8 model")
            st.image(img, caption="Segmented image", channels="BGR")

            detected_classes = [item['class'] for item in result_list_json]
            class_fq = Counter(detected_classes)
            
            
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
            st.write("Class Frequency:")
            st.dataframe(df_fq)  
 

# Video & Live cam section
if source_index == 1:

    st.header("Video Detections using YOLOv8c Dark Detector")
    video_file = st.file_uploader("Upload a video", type=["mp4"])
    process_video_button = st.button("Process Video")
    if video_file is None and process_video_button:
        st.warning("Please upload a video file to be processed!")
    if video_file is not None and process_video_button:
        with st.spinner(text='Detecting with 💕...'):
            tracker = DeepSort(max_age=5)
            centers = [deque(maxlen=30) for _ in range(10000)]
            with open(video_file.name, "wb") as f:
                f.write(video_file.read())
            video_file_out, result_video_json_file = video_processing(video_file.name, model, tracker=tracker, centers=centers)
            os.remove(video_file.name)
            st.write("Processing video...")
            st.write(video_file_out)
            st.video(video_file_out)
            with open(result_video_json_file, "r") as f:
                result_json = json.load(f)
            st.json(result_json)
    

if source_index == 2:
    st.header("Live Stream Processing using YOLOv8")
    tab_webcam = st.tabs(["Webcam Detections"])
    p_time = 0

    st.sidebar.title('Settings')
    # Choose the model
    model_type = "YOLOv8"
    sample_img = cv2.imread('assets/detective.png')
    FRAME_WINDOW = st.image(sample_img)
    cap = None


    if not model_type == 'YOLO Model':
        if model_type == 'YOLOv8':
            # GPU
            gpu_option = st.sidebar.radio(
                'Choose between:', ('CPU', 'GPU'))
            # Model
            if gpu_option == 'CPU':
                model = model
            if gpu_option == 'GPU':
                model = model

        

        # Load Class names
        class_labels = model.names


        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        # Web-cam
        
        cam_options = st.selectbox('Webcam Channel',
                                        ('Select Channel', '0', '1', '2', '3'))
    
        if not cam_options == 'Select Channel':
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(int(cam_options))
            if (cap != None) and pred:
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                tracker = DeepSort(max_age=5)
                centers = [deque(maxlen=30) for _ in range(10000)]
                frame_cnt = 0  
                
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f" NOT working\nCheck {cam_options} properly!!",
                            icon="🚨"
                        )
                        break

                    # Call DeepSort for tracking
                    img, result_list_json = image_processing(img, model, image_viewer=view_result_default, tracker=tracker, centers=centers)

                    # # Call get_frame_output to overlay distance information
                    processed_frame = get_live_frame_output(img, result_list_json)
                  
                    # Display the processed frame
                    FRAME_WINDOW.image(processed_frame, channels='BGR')
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                        
                   
                    detected_classes = [item['class'] for item in result_list_json]
                    class_fq = Counter(detected_classes)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])

                    # Updating Inference results
                    get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)             
                    frame_cnt += 1


                    


if source_index == 3:
    st.header("Live Stream Processing using YOLOv8c Dark Detector")
    tab_rtsp = st.tabs(["RTSP Detections"])
    p_time = 0

    st.sidebar.title('Settings')
    # Choose the model
    model_type = "YOLOv8"
    sample_img = cv2.imread('detective.png')
    FRAME_WINDOW = st.image(sample_img, channels='BGR')
    cap = None

    if not model_type == 'YOLO Model':
        
        if model_type == 'YOLOv8':
            # GPU
            gpu_option = st.sidebar.radio(
                'Choose between:', ('CPU', 'GPU'))

        
            # Model
            if gpu_option == 'CPU':
                model = model
                # model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = model
                # model = custom(path_or_model=path_model_file, gpu=True)

        

        # Load Class names
        class_labels = model.names


        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        


        
        rtsp_url = st.text_input(
            'RTSP URL:',
            'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
        )
        pred1 = st.checkbox(f'Predict Using rtsp {model_type}')
        cap = cv2.VideoCapture(rtsp_url)

        if (cap != None) and pred1:
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                
                tracker = DeepSort(max_age=5)
                centers = [deque(maxlen=30) for _ in range(10000)]

                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f" NOT working\nCheck {cam_options} properly!!",
                            icon="🚨"
                        )
                        break

                    
                    # Call DeepSort for tracking
                    img, result_list_json = image_processing(img, model, image_viewer=view_result_default, tracker=tracker, centers=centers)

                    # Call get_frame_output to overlay distance information
                    processed_frame = get_live_frame_output(img, result_list_json)
                    
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    # Display the processed frame
                    FRAME_WINDOW.image(processed_frame_rgb, channels='RGB')
                   
                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                        
                    # Current number of classes
                    detected_classes = [item['class'] for item in result_list_json]
                    class_fq = Counter(detected_classes)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                   
                        # Updating Inference results
                    get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
                                    
                    


class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.frame_count = 0
        self.tracker = DeepSort(max_age=5)  # Initialize the DeepSort tracker
        self.centers = [deque(maxlen=30) for _ in range(10000)]  # Initialize centers deque

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame using image_processing
        img, result_list_json = image_processing(img, model, image_viewer=view_result_default, tracker=self.tracker, centers=self.centers)
        
        # Call get_frame_output to overlay distance information
        processed_frame = get_live_frame_output(img, result_list_json)
        
        return processed_frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        new_image = self.transform(frame)
        return av.VideoFrame.from_ndarray(new_image, format="bgr24")


# Streamlit application
if source_index == 4:
    st.header("Live Stream Processing using YOLOv8c Dark Detector")
    webcam_st = st.tabs(["St webcam"])
    p_time = 0

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    count = st_autorefresh(interval=4500, limit=1000000, key="fizzbuzzcounter")
    try:
      webrtc_streamer(
        key="test",
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoTransformer
    )
    except Exception as e:
      st.error(f"Error initializing WebRTC: {e}")
   
    st.cache_data.clear()
