import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io

st.set_page_config(page_title="Raman data plotting", page_icon="microscope", layout="wide")

st.header(":microscope: Raman data plotting")

# define the actual plot function
def plot_raman(datasheet_list):
    plt.figure(figsize=(10, 6))
    
    offset = 0  # initial offset value
    
    for df, label in datasheet_list:
        plt.plot(df['Wavenumber'], df['Intensity'] + offset, label=label)
        
        # Calculate the y-axis limits to lead focus on the peaks for the current dataset with added multiplier to 
        # reach headspace above the highest peak
        intensity_max = df['Intensity'].max()
        y_upper_limit = intensity_max * 1.2  
        
        plt.ylim(0, y_upper_limit + 1100)  
        
        offset += 150  # increase offset for the next dataset
        
    plt.xlabel(f'Wavenumber ($cm^-1$)')
    plt.ylabel(f'Intensity (a.u.)')
    plt.xlim(90, 2500) 
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=True)
    plt.legend(loc='best', bbox_to_anchor=(1.0, 1.0), ncol=2, fancybox=True, shadow=True)
    plt.grid(False)
    st.pyplot(plt)

# make file uploader only accepting 'txt' files, but multiple files is fine
# check if files are uploaded
# make a datasheet list with all the dataframes from the uploaded files
uploaded_files = st.file_uploader("Upload Raman data files as .txt", type=["txt"], accept_multiple_files=True)

if not uploaded_files:
    st.write("No files uploaded.")
else:
    datasheet_list = []
    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read()
        df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')), sep="\t", header=None, names=['Wavenumber', 'Intensity'])
        label = uploaded_file.name[:-4]  # label without extension
        datasheet_list.append((df, label))

    plot_raman(datasheet_list)

    # input the file name for saving the final graph
    name_of_file = st.text_input("Enter file name for saving:")
    st.write('Your files will be saved to your Downloads folder as:', name_of_file)

    # save image as bytes to make it downloadable
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png", dpi=1200)
    # make a download button
    btn = st.download_button(
        label="Download image",
        data=img_bytes.getvalue(),
        file_name=f'{name_of_file}.png',  
        mime="image/png")
