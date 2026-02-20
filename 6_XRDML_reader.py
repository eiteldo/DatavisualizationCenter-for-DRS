import os
import pandas as pd
import xrayutilities as xu
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import io
import xml.etree.ElementTree as ET
import numpy as np


st.set_page_config(page_title="📁 Plotting X-Ray diffraction data", page_icon="📁", layout="wide")

st.header("📁 Plotting X-Ray diffraction data")

uploaded_file = st.file_uploader("Choose a file", type=["xrdml"])

# Input file name for saving
name_of_file = st.text_input("Enter file name for saving:")
st.write('Your files will be saved to your Downloads folder as:', name_of_file)

if uploaded_file is None:
    st.stop()

# make temporary file to save the uploaded content
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file_path = temp_file.name
temp_file.write(uploaded_file.read())
temp_file.close()

# Load XRDML file as xml to extract counting time
xrdml_root = ET.parse(temp_file_path).getroot()

# Load XRDML file as xml to extract counting time
namespace = "{http://www.xrdml.com/XRDMeasurement/2.1}"
commonCountingTime_element = xrdml_root.find(f".//{namespace}dataPoints/{namespace}commonCountingTime")
if commonCountingTime_element is not None:
    commonCountingTime_value = commonCountingTime_element.text
    counting_time = float(commonCountingTime_value)
else:
    st.write('Counting Time not found!')  


# Load XRDML data 
tt, inte = xu.io.getxrdml_scan(temp_file_path)

# Close and remove the temporary file
os.remove(temp_file_path)

intensity = inte*counting_time

data = pd.DataFrame({"2Theta": tt, 'Intensity': intensity})
st.write(data)

# Plot the data
final_fig = plt.figure(figsize=(10, 6))
plt.plot(tt, intensity, linewidth=2)
plt.xlabel("2Theta")
plt.ylabel("Intensity (a.u.)")
plt.tick_params(left=True, right=False, labelleft=True, labelbottom=True)
plt.xlim(min(tt), max(tt))
st.pyplot(final_fig)

# Save the figure as a bytes object
img_bytes = io.BytesIO()
plt.savefig(img_bytes, format="png", dpi=1200)

# Create a download button
btn = st.download_button(
    label="Download image",
    data=img_bytes.getvalue(),
    file_name=f'{name_of_file}.png',
    mime="image/png")





