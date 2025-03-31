import streamlit as st


st.set_page_config(layout="wide")

pages = [
    st.Page("pages/home.py", title="Game Tracker"),
    st.Page("pages/games.py", title="Stored Games"),
    st.Page("pages/settings.py", title="Settings"),
]

pg = st.navigation(pages)
pg.run()

with st.sidebar:
    st.header("PROJECT NAME")
    st.caption("A Realtime Chess Game Tracker")