import streamlit as st
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(layout="wide")

pages = [
    st.Page("pages/home.py", title="Game Tracker"),
    st.Page("pages/settings.py", title="Settings"),
]

pg = st.navigation(pages)
pg.run()

with st.sidebar:
    st.header("PROJECT NAME")
    st.caption("A Realtime Chess Game Tracker")