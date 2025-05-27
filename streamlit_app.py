import streamlit as st
import os
import datetime
import state

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

# Configure pages
st.set_page_config(layout="wide")

pages = [
    st.Page("pages/home.py", title="Game Tracker"),
    st.Page("pages/settings.py", title="Settings"),
]

pg = st.navigation(pages)
pg.run()

# Configure sidebar
with st.sidebar:
    st.header("REALTIME CHESS TRACKER")
    st.caption("A neural network assisted system to track your chess games")

    st.subheader("Game Recording")
    if st.button("Calibrate", use_container_width=True):
        state.mode = "calibrate"
    if st.button("Record", use_container_width=True):
        state.mode = "record"
    
    st.subheader("Game Settings")
    if st.button("Copy PGN", use_container_width=True):
        print("PGN copied")
    if st.button("Copy FEN", use_container_width=True):
        print("FEN copied")
    if st.button("Game Settings", use_container_width=True):
        game_settings()