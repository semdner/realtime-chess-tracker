import streamlit as st
import os

if "mode" not in st.session_state:
    st.session_state.mode = None

st.set_page_config(layout="wide")

pages = [
    st.Page("pages/home.py", title="Game Tracker"),
    st.Page("pages/settings.py", title="Settings"),
]

pg = st.navigation(pages)
pg.run()

with st.sidebar:
    st.header("REALTIME CHESS TRACKER")
    st.caption("A neural network assisted system to track your chess games")
    st.subheader("Camera Preview")
    if st.button("None", use_container_width=True):
        st.session_state.mode = None
        st.rerun()
    if st.button("Bounding Box", use_container_width=True):
        st.session_state.mode = "box"
        st.rerun()
    if st.button("Segmentation", use_container_width=True):
        st.session_state.mode = "segmentation"
        st.rerun()
    if st.button("Contour", use_container_width=True):
        st.session_state.mode = "contour"
        st.rerun()
    st.subheader("Game Recording")
    if st.button("Calibrate", use_container_width=True):
        st.session_state.mode = "calibrate"
        st.rerun()
    if st.button("Record", use_container_width=True):
        st.session_state.mode = "calibrate"
        st.rerun()
    st.subheader("Game Settings")
    if st.button("Copy PGN", use_container_width=True):
        print("PGN copied")
    if st.button("Copy FEN", use_container_width=True):
        print("FEN copied")
    if st.button("Game Settings", use_container_width=True):
        print("Game Settings")