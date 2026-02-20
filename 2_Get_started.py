import streamlit as st
from PIL import Image

st.set_page_config(page_title="Get started", page_icon="🎓", layout='wide')

st.markdown('''
    **If you have problems running any of the applcations or you do not get the expected result, check this documentation.
    For further help, use the contact form from the 'Contact' page.**
    ''')

# explanation on how to use the UV-Vis Tool

st.header("How to use the UV-Vis tool")


st.markdown('''
    When you run the application, upload the file you want to use. The file should have the following layout:
    ''')

# use expanders with images inside for keeping the page clean

with st.expander("See image"):
    image = Image.open('file_drs.png')
    st.image(image, caption=None, use_column_width='always')

st.markdown('''
    After uploading your file, you can see the imported dataframe with your data. You can use this to check if the imported data ist the corret format.
    Type in the value for calculating the transition of your semiconductor:
    - for direct allowed transition type: 2.0
    - for indirect allowed transition type: 0.5
         
    **Forbidden transition types have not been tested!**
                      
    Type the file name for your saved files. The necessary file extension will be added automatically!
    ''')

with st.expander("See image"):  
    image = Image.open('df_drs.png')
    st.image(image, caption=None, use_column_width='always')

st.markdown('''Now you can see the actual graph of your modified data. Use the sliders below to select the fitting areas for the Tauc plot and for the y-offset plot.
            As a result, you can see the selected areas in the next graph.''')

with st.expander("See image"):
    image = Image.open('select_slider_drs.png')
    st.image(image, caption=None, use_column_width='always')


st.markdown('''If you have selected you data areas, you can click the 'Start calculation' button.
This will start your calculations and will give you the final plot and values as an output.
You can then download the graph using the 'Download image' button.
If you need the values, just re-run the calculation and download the text file.''')

with st.expander("See image"):
    image = Image.open('results_drs.png')
    st.image(image, caption=None, use_column_width='always')

# Explanation on how to use the Raman plotter

st.header("How to use the Raman plotter")


st.markdown('''
    When you run the application, upload a single or multiple files you want to use. The file should have the following layout:
    ''')

with st.expander("See image"):
    image = Image.open('file_raman.png')
    st.image(image, caption=None, use_column_width='always')

st.markdown('''
    After uploading your file, you can see the displayed graph from your data. Type a file name if you want to download the graph.
    For downloading the image use the 'Download image' button.
    ''')

with st.expander("See image"):  
    image = Image.open('graph_and_button_raman.png')
    st.image(image, caption=None, use_column_width='always')

