import streamlit as st
import cv2
import torch
import numpy as np
import av
import time
import state
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
from setup_game import calibrate, init_board
from record_game import record, generate_image
from streamlit_autorefresh import st_autorefresh

st.title("Realtime Chess Tracker")

st_autorefresh(interval=2000, limit=None, key="refresh")

def camera_preview(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")

    if state.mode == "calibrate":
        state.is_calibrated = calibrate(img)
        if state.is_calibrated:
            print("calibrated correctly")
            state.board = init_board()
            state.image_path = "media/starting_board.png"
        else:
            print("not calibrated correctly")
            state.image_path = "media/empty_board.png"

        state.mode = None
    
    elif state.mode == "record":
        if state.is_calibrated:
            print("start recording")
            is_valid_move = record(img)
            if is_valid_move:
                print(state.board)
                state.image_path = generate_image(state.board)
        else:
            print("not calibreted yet")
    
    elif state.mode == None:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# tab1, tab2 = st.tabs(["Live Capture", "Video Upload"])
col1, col2 = st.columns([2, 1])


with col1:
    st.header("Camera Preview")
    webrtc_streamer(key="camera", 
        video_frame_callback=camera_preview,
        sendback_audio=False
        )
with col2:
    st.header("Game State")

    image_path = state.image_path

    # update position
    try:
        image = Image.open(image_path)
        st.image(image)
    except FileNotFoundError:
        st.error(f"Bild nicht gefunden: {image_path}")