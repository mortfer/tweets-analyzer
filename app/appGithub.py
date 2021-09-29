import appIbai
import appNetflix
import streamlit as st
#Solución de streamlit para tener varios scripts en una página web. Lo utilizo para mis dos ejemplos donde 
#sólo cambio los datos utilizados
st.set_page_config(page_title="TFM_Marc",layout="wide",initial_sidebar_state="expanded")

PAGES = {
    "IbaiLlanos": appIbai,
    "Netflix": appNetflix
}
selection = st.sidebar.radio("Ir a", list(PAGES.keys()))
page = PAGES[selection]
page.app()