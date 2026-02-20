import streamlit as st
from PIL import Image

st.set_page_config(page_title="Welcome", page_icon="👋", layout='wide')

st.header("👋 Welcome to the data-center for visualizing and fitting your data!")




st.markdown( '''
    This site does contain apps for fitting and visualizing your data. For now, you can fit and plot your UV-Vis data as well as Raman data.
    In the future other options will be added.
    
    ### Need some more information?
            
    - Be patient while the calculations are running!
    - If you want to download images and text files, download the first and run the calculation again, then download the second. 
    The results will be the same unless you change your input variables!
    - Contact me using the Contact page in the sidebar and help to make the apps better!'''
            )

st.markdown( '''### This is what your UV-Vis data plot and fit should look like:''')
            
image = Image.open('example_plot.png')

st.image(image, caption='UV-Vis data fitted and plotted using this software', use_column_width='always')

st.markdown( '''### This is what your Raman data plot should look like:''')
            
image = Image.open('example_plot_raman.png')

st.image(image, caption='Raman data plotted using this software', use_column_width='always')