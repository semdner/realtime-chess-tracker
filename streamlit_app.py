import streamlit as st
import os
import datetime

@st.dialog("Set Game Information")
def game_settings():
    with st.form("game_settings"):
        white_col, black_col = st.columns([1, 1])

        with white_col:
            white_name = st.text_input(label="White Player Name")
            white_team = st.text_input(label="White Player Team")
            white_elo  = st.text_input(label="White Player Elo")
        with black_col:
            black_name = st.text_input(label="Black Player Name")
            black_team = st.text_input(label="Black Player Team")
            black_elo  = st.text_input(label="Black Player Elo")

        date = st.date_input(label="Date when Game was played")

        if st.form_submit_button("Save"):
            st.session_state.vote = {
                "white_name": white_name, "white_team": white_team, "white_elo": white_elo,
                "black_name": black_name, "black-team": black_team, "black_elo": black_elo,
                "date": date
            }
            st.rerun()




if "mode" not in st.session_state:
    st.session_state.mode = None

if "contour" not in st.session_state:
    st.session_state.contour = None

if "corners" not in st.session_state:
    st.session_state.corners = None

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
        st.session_state.mode = "contour"
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
        game_settings()