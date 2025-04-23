import streamlit as st
import cv2
import torch
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

st.title("Realtime Chess Tracker")

model = YOLO("model/best.pt")

mode = st.session_state.mode

contour = None
corners = None

def video_model(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")

    if mode == "box":
        results = model.predict(img)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    elif mode == "segmentation":
        results = model.predict(img)
        for result in results:
           for index, item in enumerate(result):
                b_mask = np.zeros(img.shape[:2], np.uint8)
                contour = item.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

                overlay = np.zeros_like(img)

                cv2.drawContours(overlay, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)

                # Blend the overlay with the original frame
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)    
    elif mode == "contour":
            results = model.predict(img)
            for result in results:
                for index, item in enumerate(result):
                    b_mask = np.zeros(img.shape[:2], np.uint8)
                    pred_contour = item.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    new_img = cv2.drawContours(b_mask, [pred_contour], -1, (255, 255, 255), cv2.FILLED)

                    # Find contour from mask
                    mask_contours, hierarchy = cv2.findContours(new_img, 1, 2)

                    for cnt in mask_contours:
                        epsilon = 0.01 * cv2.arcLength(cnt, True)
                        contour = cv2.approxPolyDP(cnt, epsilon, True)
                        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)

                        if len(contour) == 4:
                            x1, y1 = contour[0][0][0], contour[0][0][1]
                            x2, y2 = contour[1][0][0], contour[1][0][1]
                            x3, y3 = contour[2][0][0], contour[2][0][1]
                            x4, y4 = contour[3][0][0], contour[3][0][1]
                            x1, y1 = int(x1), int(y1)
                            x2, y2 = int(x2), int(y2)
                            x3, y3 = int(x3), int(y3)
                            x4, y4 = int(x4), int(y4)
                            corners = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                            cv2.circle(img, (x1, y1), radius=5, color=(255, 0, 0), thickness=-1)
                            cv2.circle(img, (x2, y2), radius=5, color=(255, 0, 0), thickness=-1)
                            cv2.circle(img, (x3, y3), radius=5, color=(255, 0, 0), thickness=-1)
                            cv2.circle(img, (x4, y4), radius=5, color=(255, 0, 0), thickness=-1)

    elif mode == "none":
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# tab1, tab2 = st.tabs(["Live Capture", "Video Upload"])
col1, col2 = st.columns([2, 1])


with col1:
    st.header("Camera Preview")
    webrtc_streamer(key="streamer", 
        video_frame_callback=video_model,
        sendback_audio=False)
with col2:
    st.header("Game")
    st.image("lichess.png")