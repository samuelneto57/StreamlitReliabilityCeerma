# Import libraries
from turtle import width
import streamlit as st
from PIL import Image

import show_parametricmodel , show_parametricmix, show_otherfunc,\
show_comingsoon, show_repairable, show_alt, show_fitter, show_reliabilitytest, \
show_mixture, show_competingrisk

# Page layout
image_ufpe = Image.open('./src/logo.png')
image_pip = Image.open('./src/logopip.png')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
#ReportStatus {visibility: hidden;}

</style>

"""

st.set_page_config(page_title="Reliability",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

st.sidebar.image(image_ufpe)
st.sidebar.write("")

st.sidebar.title("📈 Reliability app")
st.sidebar.write("")


st.sidebar.write("""This app is an easy-to-use interface built in Streamlit
for reliability related analysis and visualization using the Reliability Python library.
""")
st.sidebar.write("")

# 👀 〽️ 🏃 🔧 🔍 💪
# Module selection
first_menu = st.sidebar.selectbox(
    "Which module do you want to use?",
    ("Select a module","Distribution Visualization", "Fit Distribution to Data", 
    "Accelerated life testing", "Repairable Systems"," Reliability Testing", 
    "Stress and Strentgh")
)

if first_menu == "Distribution Visualization":
    add_selectbox = st.sidebar.selectbox(
        "Which submodule do you want to use?",
        ("Select a submodule", "Parametric Distributions", "Mixture Model","Competing Risk Model", "Non-Parametric Model")
    )

    if add_selectbox == "Select a module":
        pass
    if add_selectbox == "Parametric Distributions":
        show_parametricmodel.show()
    # if add_selectbox == "Parametric Mix Model":
    #     show_parametricmix.show()
    if add_selectbox == 'Mixture Model':
        show_mixture.show()
    if add_selectbox == 'Competing Risk Model':
        show_competingrisk.show()
    if add_selectbox == "Non-Parametric model":
        show_comingsoon.show()

if first_menu == "Fit Distribution to Data":
    show_fitter.show()


if first_menu == "Accelerated life testing":
    show_alt.show()


if first_menu == "Repairable Systems":
    show_repairable.show()

        
if first_menu == "Reliability Testing":
    show_reliabilitytest.show()


if first_menu == "Stress and Strentgh":
    show_otherfunc.show()

# Authors name and link
# authors_css = """
#         style='
#         display: block;
#         margin-bottom: 0px;
#         margin-top: 0px;
#         padding-top: 0px;
#         font-weight: 400;
#         font-size:1.1em;
#         filter: brightness(85%);
#         text-align: center;
#         text-decoration: none;
#         '
# """
#color:#DBBD8A;

# st.sidebar.markdown(
#     '<p ' + authors_css + '>' + 'By </p>',
#         unsafe_allow_html=True
# )
# st.sidebar.markdown(f"\n\n\n")
# st.sidebar.markdown(
#     '<a ' + authors_css + ' target="_blank" href="https://github.com/biasalesc">' + 'Beatriz Cunha</a>',
#     unsafe_allow_html=True,
# )
# st.sidebar.markdown(
#     '<a ' + authors_css + ' target="_blank" href="https://github.com/yop2yop">' + 'Diego</a>',
#     unsafe_allow_html=True,
# )