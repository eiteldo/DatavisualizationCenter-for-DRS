import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# page config 
st.set_page_config(
    page_title="UV-Vis analysis tool",
    page_icon="📈",
    layout="wide",
)

st.title("📈 UV-Vis analysis tool")

# build file uploader and make dataframe from uploaded file
uploaded_file = st.file_uploader("Choose a file")

  if uploaded is None:
        st.info("Upload a TEM image to begin analysis.")
        return

data = pd.read_csv(uploaded_file, sep=';', skiprows=1)
st.write(data)

# initial data contains , decimals, that will be replaced with .
data['nm'] = data['nm'].replace(',','.',regex=True)
data['f(R)'] = data['f(R)'].replace(',','.',regex=True)

# converting the data type to float values
data['nm'] = data['nm'].astype(float)
data['f(R)'] = data['f(R)'].astype(float)

# cclculate energy in eV from imported wavelength 
energy = 1.2398 / (data['nm'].values / 1000)

# input type of transition without changes, the initial value of 0.5 will be selected (indirect allowed transition)
transition = st.number_input("Enter the type of transition:", value=0.5)

# input file name for saving the final graph and text data exported
name_of_file = st.text_input("Enter file name for saving:")
st.write('Your files will be saved to your Downloads folder as:', name_of_file)

# calculate the modified_function calculating the tauc relevant dataset from f(R) values
# show modified dataframe for correction purpose 
modified_function = (data['f(R)'].values * energy)**(transition)
new_data = pd.DataFrame({'Energy': energy, 'Modified_function': modified_function})

# for data going below 0, the mofified_function will show NaN values, that will be filled with 0
# show new dataframe with filled NaN values
new_data = new_data.fillna(0)
st.write(new_data)

st.markdown('**Select the fitting areas for both the Tauc fit and the y-offset fit using the sliders below the graph.**')

# plot the initial figure for making fitting area selection 
plt.figure(figsize=(10, 6))
plt.scatter(energy, modified_function)

plt.xticks(np.arange(1.0, 7.0, 0.25))
plt.ylim([0, 10])
plt.xlim([1.6, 6.2])
plt.grid(True)
st.pyplot(plt)

# create slider for Tauc fit range
# create slider for y-offset fit range
tauc_fit_range = st.slider(
    "Range for Tauc fit",
    min_value=float(min(energy)),
    max_value=float(max(energy)),
    value=[float(min(energy)), float(max(energy))],
    step=0.01,
    key='tauc_fit_range_key'
)

y_offset_fit_range = st.slider(
    "Range for y-offset fit",
    min_value=float(min(energy)),
    max_value=float(max(energy)),
    value=[float(min(energy)), float(max(energy))],
    step=0.01,
    key='y_offset_fit_range_key'
)

# define fuction for updating the plot acording to the input values taken from the sliders
def update_and_display_plot(transition, tauc_fit_range, y_offset_fit_range):
    modified_function = (data['f(R)'].values * energy)**(transition)
    plt.figure(figsize=(10, 6))
    plt.scatter(energy, modified_function)
    plt.xlabel("Energy (eV)")
    plt.ylabel(fr"$(f(R)*hv)^{{{transition}}}$")
    plt.xticks(np.arange(1.0, 7.0, 0.5))
    plt.ylim([0, 10])
    plt.xlim([1.6, 6.2])
    plt.axvline(x=tauc_fit_range[0], color='orange', linestyle='--')
    plt.axvline(x=tauc_fit_range[1], color='orange', linestyle='--')
    plt.axvline(x=y_offset_fit_range[0], color='green', linestyle='--')
    plt.axvline(x=y_offset_fit_range[1], color='green', linestyle='--')
    st.pyplot(plt)

# run update function to update the plot showing the fitting areas and the values
update_and_display_plot(transition, tauc_fit_range, y_offset_fit_range)



# calculate best Tauc fit for masked area using the slider input for the Tauc fit area
# make button to start all calculations
btn_start_calc = st.button(label='Start calculation')

if(btn_start_calc):
    # select subset size (number of points for fitting)
    subset_size = 30  
    # apply a mask to the dataset using the slider input values
    masked_data_tauc = (energy >= tauc_fit_range[0]) & (energy <= tauc_fit_range[1])

    x_tauc = energy[masked_data_tauc]
    y_tauc = modified_function[masked_data_tauc]

    y_tauc[np.isnan(y_tauc)] = 0

    num_data_points_tauc = len(x_tauc)

    # add starting parameters for r2 and best subset
    if num_data_points_tauc >= subset_size:
        best_r2_tauc = 0.0
        best_subset_tauc = None

        # iterate through all possible start positions and subset sizes
        for start_index_tauc in range(num_data_points_tauc - subset_size + 1):
            for subset_size_tauc in range(subset_size, num_data_points_tauc - start_index_tauc + 1):
                end_index_tauc = start_index_tauc + subset_size_tauc
                x_subset_tauc = x_tauc[start_index_tauc:end_index_tauc]
                y_subset_tauc = y_tauc[start_index_tauc:end_index_tauc]

                # apply the model fitting a linear regression model to the subset and calculate r2 of this fit
                model = LinearRegression().fit(x_subset_tauc.reshape((-1, 1)), y_subset_tauc)
                r2_tauc = model.score(x_subset_tauc.reshape((-1, 1)), y_subset_tauc)
                # check if the r2 of this subset is larger then the best r2 value
                if r2_tauc > best_r2_tauc:
                    best_r2_tauc = r2_tauc
                    best_subset_tauc = (start_index_tauc, end_index_tauc)
                    best_subset_size_tauc = subset_size_tauc
        # if it is, the subset will be selected and the final fit will be calculated
        if best_subset_tauc is not None:
            start_index_tauc, end_index_tauc = best_subset_tauc
            x_best_subset_tauc = x_tauc[start_index_tauc:end_index_tauc]
            y_best_subset_tauc = y_tauc[start_index_tauc:end_index_tauc]
            # change x-linespace to the whole graph 
            x_fit = np.linspace(1.6, 6.2, 100)

            model = LinearRegression().fit(x_best_subset_tauc.reshape((-1, 1)), y_best_subset_tauc)
            y_pred_tauc = model.predict(x_fit.reshape((-1, 1)))
            
            # show values of fit
            st.markdown('**Calculated values for Tauc plot:**')
            st.write('Fit from:', tauc_fit_range[0], 'to',  tauc_fit_range[1])

            st.write('Best subset R-squared:', best_r2_tauc)
            st.write('Best subset size:', best_subset_size_tauc)
            st.write('Intercept =', round(model.intercept_, 3))
            st.write('Slope =', round(np.ndarray.item(model.coef_), 3))

        else:
            st.write('No fitting subset.')
    else:
        st.write('Not enough data points for fitting.')
    # calculate slope and intercept
    slope_tauc = np.ndarray.item(model.coef_)
    intercept_tauc = model.intercept_ 



    # calculate best y-offset fit for masked area
    # programm is the same as for the tauc values
    subset_size = 50

    masked_data_m = (energy >= y_offset_fit_range[0]) & (energy <= y_offset_fit_range[1])

    x_m = energy[masked_data_m]
    y_m = modified_function[masked_data_m]

    y_m[np.isnan(y_m)] = 0

    num_data_points_m = len(x_m)

    if num_data_points_m >= subset_size:
        best_r2_m = 0.0
        best_subset_m = None

        for start_index_m in range(num_data_points_m - subset_size + 1):
            for subset_size_m in range(subset_size, num_data_points_m - start_index_m + 1):
                end_index_m = start_index_m + subset_size_m
                x_subset_m = x_m[start_index_m:end_index_m]
                y_subset_m = y_m[start_index_m:end_index_m]

                model = LinearRegression().fit(x_subset_m.reshape((-1, 1)), y_subset_m)
                r2_m = model.score(x_subset_m.reshape((-1, 1)), y_subset_m)

                if r2_m > best_r2_m:  #
                    best_r2_m = r2_m
                    best_subset_m = (start_index_m, end_index_m)
                    best_subset_size_m = subset_size_m

        if best_subset_m is not None:
            start_index_m, end_index_m = best_subset_m
            x_best_subset_m = x_m[start_index_m:end_index_m]
            y_best_subset_m = y_m[start_index_m:end_index_m]

            x_fit = np.linspace(1.6, 6.2, 100)

            model = LinearRegression().fit(x_best_subset_m.reshape((-1, 1)), y_best_subset_m)
            y_pred_m = model.predict(x_fit.reshape((-1, 1)))

            st.markdown('**Calculated values for y-offset plot:**')
            st.write('Fit from:', y_offset_fit_range[0], 'to',  y_offset_fit_range[1])
          
            st.write('Best subset R-squared:', best_r2_m)
            st.write('Best subset size:', best_subset_size_m)
            st.write('Intercept =', round(model.intercept_, 3))
            st.write('Slope =', round(np.ndarray.item(model.coef_), 3))

        else:
            st.write('No fitting subset.')
    else:
        st.write('Not enough data points for fitting.')


    slope_m = np.ndarray.item(model.coef_)
    intercept_m = model.intercept_ 

    # find x value where y_offset_fit and tauc_fit have intersection (rounded to 3 digits)
    def intersection(m1, b1, m2, b2):
        x_intersection = (b2 - b1) / (m1 - m2)
        return x_intersection

    intersection_x1 = np.around(intersection(slope_m, intercept_m, slope_tauc, intercept_tauc), 3)


    # tauc intercept with x-axis (rounded to 3 digits)
    intersection_tauc = np.around(-intercept_tauc / slope_tauc, 3)

    # show the final values and plot of the fitted data
    st.header('Results and Plots')

    # show intersection values 
    st.write('Intersection Tauc fit:', intersection_tauc, 'eV')
    st.write('Intersection Tauc fit with y-Offset:', intersection_x1, 'eV')

    # make and show the final plot
    st.subheader('Final Plot')

    final_fig = plt.figure(figsize=(10, 6))
    plt.scatter(energy, modified_function, label='Original Data', s=8)
    plt.scatter(x_best_subset_tauc, y_best_subset_tauc, color='orange', label='Selected data for Tauc fit', s=8)
    plt.scatter(x_best_subset_m, y_best_subset_m, color='green', label='Selected data for y-Offset fit', s=8)
    plt.plot(x_fit, y_pred_tauc, color='red', label= f'Tauc Fit Intersection x = {intersection_tauc}')
    plt.plot(x_fit, y_pred_m, color='red', linestyle='--', label='Linear y-Offset Fit')
    plt.axvline(x=intersection_x1, color='green', label= f'Intersection x = {intersection_x1}')
    plt.xlabel("Energy (eV)")
    plt.ylabel(fr"$(f(R)*hv)^{{{transition}}}$")
    plt.xticks(np.arange(1.0, 7.0, 0.5))
    plt.ylim([0, 10])
    plt.xlim([1.6, 6.2])
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3, fancybox=True, shadow=False)
    st.pyplot(final_fig)

    # save figure as a bytes object to make it downloadable
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png", dpi=1200)
    
    # make a download button
    btn = st.download_button(
        label="Download image",
        data=img_bytes.getvalue(),
        file_name=f'{name_of_file}',
        mime="image/png")
        
    # make a textfile with the data of the fit inside
    text_contents = f'''Intersection Tauc fit: {intersection_tauc} eV 
Intersection Tauc fit with y-Offset: {intersection_x1} eV

Tauc plot values:

Best subset R-squared: {best_r2_tauc}
Best subset size: {best_subset_size_tauc}
Intercept: {intercept_tauc}
Slope: {slope_tauc}

Y-offset plot values:

Best subset R-squared: {best_r2_m}
Best subset size: {best_subset_size_m}
Intercept: {intercept_m}
Slope: {slope_m}
'''

    st.download_button('Download values as text', text_contents, file_name=f'{name_of_file}',mime="text")

