import streamlit as st

@st.dialog("Set Game Information")
def game_settings():
    with st.form("Game Information", border=False, enter_to_submit=False):
        st.caption("Enter information about the players and the game to be included in the PGN.")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.text_input("White", placeholder="Name of the white player")
            st.number_input("White Elo", value=0)
            st.text_input("White Team", placeholder="Team of the white player")

        with info_col2:
            st.text_input("Black", placeholder="Name of the black player")
            st.number_input("Black Elo", value=0)
            st.text_input("Black Team", placeholder="Name of the black player")

        st.date_input("Date")

        submitted = st.form_submit_button("Save")
        if submitted:
            st.rerun()

# col1, col2 = st.columns(2)
st.camera_input("Model Predictions and Previews")
st.multiselect("Objects to display", ["Chessboard", "Corners", "Pieces"])

# with col1:
#     st.camera_input("Model Predictions and Previews")
#     st.multiselect("Objects to display", ["Chessboard", "Corners", "Pieces"])

# with col2:
#     st.image("src/transformed2.png")

#     btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
#     with btn_col1:
#         if st.button("Copy FEN"):
#             st.success("FEN copied to clipboard.")

#     with btn_col2:
#         if st.button("Copy PGN"):
#             st.success("PGN copied to clipboard.")

#     with btn_col3:
#         if "game_settings" not in st.session_state:
#             if st.button("Game Settings"):
#                 game_settings()

#     with btn_col4:
#         st.button("Save Game")
