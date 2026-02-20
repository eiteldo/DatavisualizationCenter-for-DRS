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

# file upload
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is None:
    st.info("Upload a UV-Vis data file to begin analysis.")
    st.stop()

# Load & clean data 
data = pd.read_csv(uploaded_file, sep=';', skiprows=1)
st.write(data)

for col in ['nm', 'f(R)']:
    if data[col].dtype == object:
        data[col] = data[col].str.replace(',', '.', regex=False)
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Calculate photon energy (eV) from wavelength (nm)
energy = (1.2398 / (data['nm'].values / 1000))  # returns numpy array

# User inputs 
transition = st.number_input(
    "Enter the type of transition (e.g. 0.5 = indirect allowed):",
    value=0.5, step=0.5
)

name_of_file = st.text_input("Enter file name for saving (without extension):")
st.write('Your files will be saved as:', name_of_file)

# Compute Tauc modified function
def compute_modified(f_r, energy_arr, n):
    result = (f_r * energy_arr) ** n
    result = np.where(np.isfinite(result), result, 0.0)
    return result

modified_function = compute_modified(data['f(R)'].values, energy, transition)

new_data = pd.DataFrame({'Energy (eV)': energy, 'Modified function': modified_function})
st.write(new_data)

# Initial scatter plot
st.markdown('**Select fitting ranges using the sliders below the graph.**')

fig_preview, ax_preview = plt.subplots(figsize=(10, 6))
ax_preview.scatter(energy, modified_function, s=6)
ax_preview.set_xticks(np.arange(1.0, 7.0, 0.25))
ax_preview.set_ylim([0, 10])
ax_preview.set_xlim([1.6, 6.2])
ax_preview.grid(True)
st.pyplot(fig_preview)
plt.close(fig_preview)

# Range sliders
e_min, e_max = float(energy.min()), float(energy.max())

tauc_fit_range = st.slider(
    "Range for Tauc fit",
    min_value=e_min, max_value=e_max,
    value=(e_min, e_max), step=0.01,
    key='tauc_fit_range_key'
)

y_offset_fit_range = st.slider(
    "Range for y-offset fit",
    min_value=e_min, max_value=e_max,
    value=(e_min, e_max), step=0.01,
    key='y_offset_fit_range_key'
)

# Preview plot with fit-range markers
fig_ranges, ax_ranges = plt.subplots(figsize=(10, 6))
ax_ranges.scatter(energy, modified_function, s=6)
ax_ranges.set_xlabel("Energy (eV)")
ax_ranges.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
ax_ranges.set_xticks(np.arange(1.0, 7.0, 0.5))
ax_ranges.set_ylim([0, 10])
ax_ranges.set_xlim([1.6, 6.2])
for x in tauc_fit_range:
    ax_ranges.axvline(x=x, color='orange', linestyle='--', label='Tauc range')
for x in y_offset_fit_range:
    ax_ranges.axvline(x=x, color='green', linestyle='--', label='y-offset range')
ax_ranges.grid(True)
st.pyplot(fig_ranges)
plt.close(fig_ranges)

# best linear subset by R2
def best_linear_subset(x_arr, y_arr, min_subset):
    """Return (model, best_r2, subset_size, x_subset, y_subset) or None."""
    n = len(x_arr)
    if n < min_subset:
        return None

    best_r2 = -np.inf
    best = None

    for start in range(n - min_subset + 1):
        for size in range(min_subset, n - start + 1):
            end = start + size
            xb, yb = x_arr[start:end], y_arr[start:end]
            m = LinearRegression().fit(xb.reshape(-1, 1), yb)
            r2 = m.score(xb.reshape(-1, 1), yb)
            if r2 > best_r2:
                best_r2 = r2
                best = (m, r2, size, xb, yb)

    return best  # (model, r2, size, x_subset, y_subset)

# Calculation
btn_start_calc = st.button(label='Start calculation')

if btn_start_calc:

    # Recompute with current transition in case it changed after initial render
    modified_function = compute_modified(data['f(R)'].values, energy, transition)

    x_fit = np.linspace(1.6, 6.2, 100)

    # Tauc fit
    mask_tauc = (energy >= tauc_fit_range[0]) & (energy <= tauc_fit_range[1])
    # BUG FIX 5: use .copy() so index-based slicing on numpy array is clean
    x_tauc = energy[mask_tauc].copy()
    y_tauc = modified_function[mask_tauc].copy()
    y_tauc[~np.isfinite(y_tauc)] = 0.0

    result_tauc = best_linear_subset(x_tauc, y_tauc, min_subset=30)

    if result_tauc is None:
        st.error("Tauc fit failed: not enough data points or no valid subset found.")
        st.stop()

    model_tauc, r2_tauc, size_tauc, x_best_tauc, y_best_tauc = result_tauc
    y_pred_tauc = model_tauc.predict(x_fit.reshape(-1, 1))
    slope_tauc = float(model_tauc.coef_[0])
    intercept_tauc = float(model_tauc.intercept_)

    st.markdown('**Calculated values for Tauc plot:**')
    st.write(f'Fit range: {tauc_fit_range[0]:.3f} – {tauc_fit_range[1]:.3f} eV')
    st.write(f'Best subset R²: {r2_tauc:.6f}')
    st.write(f'Best subset size: {size_tauc}')
    st.write(f'Intercept = {intercept_tauc:.3f}')
    st.write(f'Slope = {slope_tauc:.3f}')

    # y-offset fit
    mask_m = (energy >= y_offset_fit_range[0]) & (energy <= y_offset_fit_range[1])
    x_m = energy[mask_m].copy()
    y_m = modified_function[mask_m].copy()
    y_m[~np.isfinite(y_m)] = 0.0

    result_m = best_linear_subset(x_m, y_m, min_subset=50)

    if result_m is None:
        st.error("y-offset fit failed: not enough data points or no valid subset found.")
        st.stop()

    model_m, r2_m, size_m, x_best_m, y_best_m = result_m
    y_pred_m = model_m.predict(x_fit.reshape(-1, 1))
    slope_m = float(model_m.coef_[0])
    intercept_m = float(model_m.intercept_)

    st.markdown('**Calculated values for y-offset plot:**')
    st.write(f'Fit range: {y_offset_fit_range[0]:.3f} – {y_offset_fit_range[1]:.3f} eV')
    st.write(f'Best subset R²: {r2_m:.6f}')
    st.write(f'Best subset size: {size_m}')
    st.write(f'Intercept = {intercept_m:.3f}')
    st.write(f'Slope = {slope_m:.3f}')

    # Intersection calculations
    # Where Tauc line crosses x-axis (y = 0)
    intersection_tauc = round(-intercept_tauc / slope_tauc, 3)
    # Where Tauc line crosses y-offset line
    intersection_x1 = round((intercept_m - intercept_tauc) / (slope_tauc - slope_m), 3)

    st.header('Results and Plots')
    st.write(f'Tauc x-axis intercept: **{intersection_tauc} eV**')
    st.write(f'Tauc / y-offset intersection: **{intersection_x1} eV**')

    #  Final plot
    st.subheader('Final Plot')
    final_fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(energy, modified_function, s=6, label='Original data')
    ax.scatter(x_best_tauc, y_best_tauc, color='orange', s=8, label='Tauc fit data')
    ax.scatter(x_best_m, y_best_m, color='green', s=8, label='y-offset fit data')
    ax.plot(x_fit, y_pred_tauc, color='red', label=f'Tauc fit  (x-int = {intersection_tauc} eV)')
    ax.plot(x_fit, y_pred_m, color='red', linestyle='--', label='y-offset fit')
    ax.axvline(x=intersection_x1, color='green', label=f'Intersection = {intersection_x1} eV')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
    ax.set_xticks(np.arange(1.0, 7.0, 0.5))
    ax.set_ylim([0, 10])
    ax.set_xlim([1.6, 6.2])
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        mode="expand", borderaxespad=0, ncol=3, fancybox=True
    )
    st.pyplot(final_fig)

    # Downloads 
    img_bytes = io.BytesIO()
    final_fig.savefig(img_bytes, format="png", dpi=300)
    img_bytes.seek(0)
    plt.close(final_fig)

    fname = name_of_file if name_of_file else "uvvis_result"

    st.download_button(
        label="Download image (PNG)",
        data=img_bytes.getvalue(),
        file_name=f'{fname}.png',
        mime="image/png"
    )

    text_contents = f"""Intersection Tauc fit: {intersection_tauc} eV
Intersection Tauc fit with y-Offset: {intersection_x1} eV

Tauc plot values:
  Best subset R²: {r2_tauc}
  Best subset size: {size_tauc}
  Intercept: {intercept_tauc}
  Slope: {slope_tauc}

Y-offset plot values:
  Best subset R²: {r2_m}
  Best subset size: {size_m}
  Intercept: {intercept_m}
  Slope: {slope_m}
"""

    st.download_button(
        label="Download values (TXT)",
        data=text_contents,
        file_name=f'{fname}.txt',
        mime="text/plain"
    )

