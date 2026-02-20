import streamlit as st
from PIL import Image

st.set_page_config(page_title="Welcome", page_icon="👋", layout='wide')

st.header("👋 Welcome to the datacenter for visualizing and fitting your research data!")




st.markdown( '''
    This site contains small apps for fitting and visualizing research data by eliminating user errors and making work easier. For now, you can fit and plot UV-Vis data, Raman data and analyze layered structures from TEM images.
    In the future other options might be added.
    
    ### Need some more information?
            
    - Be patient while the calculations are running! Depending on the dataset, calculations might take some time.
    - Make sure your data is in an appropriate format in order to be plotted.'''
            )


