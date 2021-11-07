import appIbai
import appNetflix
import streamlit as st
#Streamlit's solution for multi-page apps. appIbai.py and appNetflix.py are identical but using different data
st.set_page_config(page_title="TFM_Marc",layout="wide",initial_sidebar_state="expanded")

PAGES = {
    "IbaiLlanos": appIbai,
    "Netflix": appNetflix
}
selection = st.sidebar.radio("Ir a", list(PAGES.keys()))
page = PAGES[selection]
page.app()