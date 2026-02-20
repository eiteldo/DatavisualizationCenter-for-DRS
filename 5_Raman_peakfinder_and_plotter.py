import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import numpy as np
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Raman peakfinder", page_icon="microscope", layout="wide")

st.header(":microscope: Raman peakfinder and plotting tool")


def plot_raman(df, label, apply_filter, window_size, peak_intensity_slider, show_peaks):
    intensity_data = df['Intensity']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if apply_filter:
        intensity_data = savgol_filter(intensity_data, window_length=window_size, polyorder=2)
    
    ax.plot(df['Wavenumber'], intensity_data, label=label)
    
    ax.set_xlabel(r'$(cm^{-1})$')
    ax.set_ylabel(f'Intensity (a.u.)')
    ax.set_xlim(90, 2500)
    plt.xticks(np.arange(90, 2600, 250))
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=True)
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, fancybox=True, shadow=False)
    ax.grid(False)

    
    peaks, _ = find_peaks(intensity_data, height=peak_intensity_slider)
    peak_wavenumbers = df['Wavenumber'][peaks]
    peak_intensities = intensity_data[peaks]

    if show_peaks:
        ax.plot(peak_wavenumbers, peak_intensities, 'o', color='red')
    

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png", dpi=1200)
    plt.close(fig) 

    st.image(img_bytes, caption=label, use_column_width=True)
    st.write(peak_wavenumbers.tolist())


uploaded_file = st.file_uploader("Upload Raman data file as .txt", type=["txt"])
apply_filter = st.checkbox("Apply Savitzky-Golay Filter", value=False)
show_peaks = st.checkbox("Show Peaks", value=True)
window_size = st.slider("Adjust Savitzky-Golay Filter Window Size", min_value=3, max_value=101, value=21, step=2)
calculate_button = st.button("Calculate Graph")

if not uploaded_file:
    st.write("No file uploaded.")
else:
    file_contents = uploaded_file.read()
    df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')), sep="\t", header=None, names=['Wavenumber', 'Intensity'])
    label = uploaded_file.name[:-4]
  
    min_intensity = float(df['Intensity'].min())
    max_intensity = float(df['Intensity'].max())
    peak_intensity_slider = st.slider(
    f"Adjust peak intensity for finding peaks",
        min_value=min_intensity,
        max_value=max_intensity,
        value=(min_intensity + max_intensity) / 2,  # Set an initial value within the range
        step=0.1,
        format="%.2f",
        key=f'peak_intensity'
    )



if calculate_button:
    plot_raman(df, label, apply_filter, window_size, peak_intensity_slider, show_peaks)

