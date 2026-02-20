import streamlit as st
from PIL import Image

st.set_page_config(page_title="Get started", page_icon="🎓", layout='wide')

st.markdown('''
    **If you have problems running any of the applcations or you do not get the expected result, check this documentation.**
    ''')

st.header("How to use the UV-Vis tool")


st.markdown('''
    When you run the application, upload the file you want to use. The file should have the following layout:
    ''')

with st.expander("See image"):
    image = Image.open('file_drs.png')
    st.image(image, caption=None, use_column_width='always')

st.markdown('''
    After uploading your file, you can see the imported dataframe with your data. You can use this to check if the imported data ist the corret format.
    Type in the value for calculating the transition of your sample, e.g.:
    
    - for direct allowed transition type: 2.0
    - for indirect allowed transition type: 0.5
                      
    Type the file name for your saved files. The necessary file extension will be added automatically! Fill in the number of datapoints the fitting algorithm has to use.
    ''')

st.markdown('''Click the 'Start calculation' button.
This will start your calculations and will give you the final plot and values as an output.
You can then download the graph and the values as as .txt using the 'Download' buttons.''')


# Explanation on how to use the TEM analysis tool

st.header("How to use the TEM analysis tool")


st.markdown('''
    When you run the application, upload a single file you want to use. The file has to be an image of the layers (graphene or similar) you want to analyze:
    ''')

with st.expander("See image"):
    image = Image.open('TEM_example_image.jpg')
    st.image(image, caption=None)

st.markdown('''
    After uploading your file, you can directly see the analyzed data from your image.
    For downloading the image use the 'Download image' button.
    ''')




