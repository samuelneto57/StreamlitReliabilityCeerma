import streamlit as st

import show_parametricmodel , show_parametricmix, show_otherfunc,show_comingsoon, show_repairable, show_alt, show_fitter
from PIL import Image

image_ufpe = Image.open('./src/logo.png')
image_pip = Image.open('./src/logopip.png')
image_ceerma = Image.open('./src/favicon.png')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
#ReportStatus {visibility: hidden;}

</style>

"""

st.set_page_config(page_title="Reliability",page_icon=image_ceerma,layout="wide", initial_sidebar_state="expanded")

st.sidebar.image(image_ufpe)
st.sidebar.write("")

st.sidebar.title("📈 Reliability app")
st.sidebar.write("")


st.sidebar.write("""This app is an easy-to-use interface built in Streamlit
for reliability related analysis and visualization using the Reliability Python library.
""")
st.sidebar.write("")

modules = (
    "Select a module",
    "Parametrics and Non-Parametrics Models", 
    "Accelerated Life Testing", 
    "Repairable Systems", 
    # "Other Functions"
    "Stress and Strength"
)

first_menu = st.sidebar.selectbox("Which module do you want to use?", modules)

if first_menu == "Parametrics and Non-Parametrics Models":

    submodules = {
        "Select a submodule": lambda: None,
        "Parametric Model": show_parametricmodel.show,
        "Parametric Mix Model": show_parametricmix.show,
        # "Non-Parametric Model": show_comingsoon.show,
        "Fitter": show_fitter.show
    }
    
    add_selectbox = st.sidebar.selectbox("Which submodule do you want to use?", 
                                         list(submodules))

    submodules[add_selectbox]()

if first_menu == "Accelerated Life Testing":
    show_alt.show()
    # add_selectbox = st.sidebar.selectbox(
    #     "Which submodule do you want to use?",
    #     ("Select a submodule", "Accelerated life testing")
    # )

    # if add_selectbox == "Select a module":
    #     pass
    # if add_selectbox == "Accelerated life testing":
    #     show_alt.show()


if first_menu == "Repairable Systems":
    show_repairable.show()
    # add_selectbox = st.sidebar.selectbox(
    #     "Which submodule do you want to use?",
    #     ("Select a submodule", "Repairable Systems")
    # )

    # if add_selectbox == "Select a module":
    #     pass
    # if add_selectbox == "Repairable Systems":
    #     show_repairable.show()

if first_menu == "Stress and Strength":
    show_otherfunc.show()
# if first_menu == "Other Functions":
#     add_selectbox = st.sidebar.selectbox(
#         "Which submodule do you want to use?",
#         ("Select a submodule", "Stress and Strentgh")
#     )
#     if add_selectbox == "Select a module":
#         pass

#     if add_selectbox == "Stress and Strentgh":
#         show_otherfunc.show()
