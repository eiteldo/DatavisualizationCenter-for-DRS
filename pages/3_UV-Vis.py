import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# page config 
st.set_page_config(page_title="UV-Vis analysis tool", page_icon="📈", layout="wide")
st.title("📈 UV-Vis analysis tool")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is None:
    st.info("Upload a UV-Vis data file to begin analysis.")
    st.stop()

# Load and clean data
data = pd.read_csv(uploaded_file, sep=';', skiprows=1)
st.write(data)

for col in ['nm', 'f(R)']:
    if data[col].dtype == object:
        data[col] = data[col].str.replace(',', '.', regex=False)
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Energy calc
energy = 1.2398 / (data['nm'].values / 1000)

transition = st.number_input(
    "Enter the type of transition (e.g. 0.5 = indirect allowed):",
    value=0.5, step=0.5
)
name_of_file = st.text_input("Enter file name for saving (without extension):")

def compute_modified(f_r, energy_arr, n):
    result = (f_r * energy_arr) ** n
    return np.where(np.isfinite(result), result, 0.0)

modified_function = compute_modified(data['f(R)'].values, energy, transition)

new_data = pd.DataFrame({'Energy (eV)': energy, 'Modified function': modified_function})
st.write(new_data)

# Initial plot 
st.markdown('**Select fitting ranges using the sliders below the graph.**')
fig0, ax0 = plt.subplots(figsize=(10, 6))
ax0.scatter(energy, modified_function, s=6)
ax0.set_xticks(np.arange(1.0, 7.0, 0.25))
ax0.set_ylim([0, 10]); ax0.set_xlim([1.6, 6.2]); ax0.grid(True)
st.pyplot(fig0); plt.close(fig0)

# Sliders
e_min, e_max = float(energy.min()), float(energy.max())
tauc_fit_range = st.slider("Range for Tauc fit", e_min, e_max,
                            (e_min, e_max), step=0.01, key='tauc')
makula_fit_range = st.slider("Range for Makula (y-offset) fit", e_min, e_max,
                              (e_min, e_max), step=0.01, key='makula')

# Preview plot with range markers
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(energy, modified_function, s=6)
ax1.set_xlabel("Energy (eV)")
ax1.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
ax1.set_xticks(np.arange(1.0, 7.0, 0.5))
ax1.set_ylim([0, 10]); ax1.set_xlim([1.6, 6.2]); ax1.grid(True)
ax1.text(1.7, 9.1,
         f'Tauc fit: {tauc_fit_range[0]:.2f}–{tauc_fit_range[1]:.2f}\n'
         f'Makula fit: {makula_fit_range[0]:.2f}–{makula_fit_range[1]:.2f}')
for x in tauc_fit_range:
    ax1.axvline(x=x, color='orange', linestyle='--')
for x in makula_fit_range:
    ax1.axvline(x=x, color='green', linestyle='--')
st.pyplot(fig1); plt.close(fig1)

# Core fitting function
def best_linear_subset(energy_full, mod_full, x_min, x_max, min_subset):
    n = len(energy_full)
    best_r2 = 0.0
    best_result = None

    for subset_size in range(min_subset, n + 1):
        for i in range(n - subset_size + 1):
            x_slice = energy_full[i : i + subset_size]
            y_slice = mod_full[i : i + subset_size]

            mask = (x_slice >= x_min) & (x_slice <= x_max)
            if mask.sum() < min_subset:
                continue

            xb, yb = x_slice[mask], y_slice[mask]

            slope, intercept = np.polyfit(xb, yb, 1)

            y_mean = np.mean(yb)
            ss_tot = np.sum((yb - y_mean) ** 2)
            if ss_tot == 0:
                continue
            ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
            r2 = 1.0 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_result = (slope, intercept, r2, int(mask.sum()), xb.copy(), yb.copy())

    return best_result

# Calculation button
if st.button(label='Start calculation'):

    modified_function = compute_modified(data['f(R)'].values, energy, transition)
    x_fit = np.linspace(1.6, 6.2, 100)

    # Tauc fit
    result_tauc = best_linear_subset(
        energy, modified_function,
        tauc_fit_range[0], tauc_fit_range[1],
        min_subset=30 #can be changed, but must not be too high
    )
    if result_tauc is None:
        st.error("Tauc fit failed: not enough data points in the selected range.")
        st.stop()

    slope_tauc, intercept_tauc, r2_tauc, size_tauc, x_best_tauc, y_best_tauc = result_tauc
    y_pred_tauc = slope_tauc * x_fit + intercept_tauc

    st.markdown('**Tauc plot fit:**')
    st.write(f'Range: {tauc_fit_range[0]:.3f} – {tauc_fit_range[1]:.3f} eV')
    st.write(f'Best subset R²: {r2_tauc:.6f} | size: {size_tauc}')
    st.write(f'Slope = {slope_tauc:.4f} | Intercept = {intercept_tauc:.4f}')

    # Makula fit
    result_m = best_linear_subset(
        energy, modified_function,
        makula_fit_range[0], makula_fit_range[1],
        min_subset=50   # can be changed
    )
    if result_m is None:
        st.error("Makula fit failed: not enough data points in the selected range.")
        st.stop()

    slope_m, intercept_m, r2_m, size_m, x_best_m, y_best_m = result_m
    y_pred_m = slope_m * x_fit + intercept_m

    st.markdown('**Makula (y-offset) plot fit:**')
    st.write(f'Range: {makula_fit_range[0]:.3f} – {makula_fit_range[1]:.3f} eV')
    st.write(f'Best subset R²: {r2_m:.6f} | size: {size_m}')
    st.write(f'Slope = {slope_m:.4f} | Intercept = {intercept_m:.4f}')

    # Intersections
    intersection_tauc   = round(-intercept_tauc / slope_tauc, 3)
    intersection_makula = round((intercept_m - intercept_tauc) / (slope_tauc - slope_m), 3)

    st.header('Results')
    st.write(f'Estimated bandgap (Tauc x-axis intercept): **{intersection_tauc} eV**  R²={r2_tauc:.4f}')
    st.write(f'Estimated bandgap (Makula intersection):   **{intersection_makula} eV**  R²={r2_m:.4f}')

    # plot
    st.subheader('Final Plot')
    final_fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(energy, modified_function, s=6, label='Data')
    ax.scatter(x_best_tauc, y_best_tauc, color='orange', s=8, label='Tauc fit data')
    ax.scatter(x_best_m,    y_best_m,    color='green',  s=8, label='Makula fit data')
    ax.plot(x_fit, y_pred_tauc, color='red',
            label=f'Tauc fit  R²={r2_tauc:.4f}  x-int={intersection_tauc} eV')
    ax.plot(x_fit, y_pred_m, color='red', linestyle='--',
            label=f'Makula fit  R²={r2_m:.4f}')
    ax.axvline(x=intersection_makula, color='green',
               label=f'Makula intersection = {intersection_makula} eV')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
    ax.set_xticks(np.arange(1.0, 7.0, 0.5))
    ax.set_ylim([0, 10]); ax.set_xlim([1.6, 6.2])
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=3, fancybox=True)
    st.pyplot(final_fig)

    # Download
    img_bytes = io.BytesIO()
    final_fig.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close(final_fig)

    fname = name_of_file if name_of_file else "uvvis_result"

    st.download_button("Download image (PNG)", data=img_bytes.getvalue(),
                       file_name=f'{fname}.png', mime="image/png")

    text_contents = f"""Estimated bandgap (Tauc x-axis intercept): {intersection_tauc} eV
Estimated bandgap (Makula intersection):   {intersection_makula} eV

Tauc fit:
  Range: {tauc_fit_range[0]} - {tauc_fit_range[1]} eV
  R2: {r2_tauc}
  Points: {size_tauc}
  Slope: {slope_tauc}
  Intercept: {intercept_tauc}

Makula fit:
  Range: {makula_fit_range[0]} - {makula_fit_range[1]} eV
  R2: {r2_m}
  Points: {size_m}
  Slope: {slope_m}
  Intercept: {intercept_m}
"""
    st.download_button("Download values (TXT)", data=text_contents,
                       file_name=f'{fname}.txt', mime="text/plain")
