import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter1d

# page config
st.set_page_config(page_title="UV-Vis analysis tool", page_icon="📈", layout="wide")
st.title("📈 UV-Vis analysis tool")

# HELPERS

def compute_modified(f_r, energy_arr, n):
    result = (f_r * energy_arr) ** n
    return np.where(np.isfinite(result), result, 0.0)


def fit_line(xb, yb):
    """np.polyfit + R² for a pair of arrays."""
    slope, intercept = np.polyfit(xb, yb, 1)
    ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
    ss_tot = np.sum((yb - yb.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def method_derivative(energy, mod, x_min, x_max, min_pts, sigma, threshold_pct):
    # 1. Crop to slider range
    mask_range = (energy >= x_min) & (energy <= x_max)
    xe = energy[mask_range]
    ye = mod[mask_range]

    if len(xe) < min_pts:
        return None

    # 2. Smooth
    ye_smooth = gaussian_filter1d(ye, sigma=sigma)

    # 3. Second derivative via central differences
    d2y = np.gradient(np.gradient(ye_smooth, xe), xe)

    # 4. Threshold: accept points where curvature is small
    threshold = (threshold_pct / 100.0) * np.max(np.abs(d2y))
    linear_mask = np.abs(d2y) < threshold

    # Find longest contiguous True run
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for k, val in enumerate(linear_mask):
        if val:
            if cur_len == 0:
                cur_start = k
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_len < min_pts:
        return None

    # 5. Fit line to the found segment (using original unsmoothed data)
    xb = xe[best_start : best_start + best_len]
    yb = ye[best_start : best_start + best_len]
    slope, intercept, r2 = fit_line(xb, yb)

    return slope, intercept, r2, best_len, xb.copy(), yb.copy()



# FILE UPLOAD

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is None:
    st.info("Upload a UV-Vis data file to begin analysis.")
    st.stop()

data = pd.read_csv(uploaded_file, sep=';', skiprows=1)
st.write(data)

for col in ['nm', 'f(R)']:
    if data[col].dtype == object:
        data[col] = data[col].str.replace(',', '.', regex=False)
    data[col] = pd.to_numeric(data[col], errors='coerce')

energy = 1.2398 / (data['nm'].values / 1000)


# INPUTS

col1, col2 = st.columns(2)
with col1:
    transition = st.number_input(
        "Type of transition (0.5 = indirect allowed, 2 = direct allowed):",
        value=0.5, step=0.5
    )
with col2:
    name_of_file = st.text_input("File name for saving (without extension):")

modified_function = compute_modified(data['f(R)'].values, energy, transition)

# Method parameters
st.markdown("---")
st.subheader("⚙️ Derivative method parameters")
st.caption(
    "The method smooths the data, computes the second derivative (curvature), "
    "and finds the longest region within your selected range where curvature is "
    "below the threshold. The line is then fit to that region using the original data."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    sigma = st.slider("Smoothing (sigma)", 1, 15, 3,
        help="Gaussian smoothing before differentiating. Increase for noisy spectra.")
with c2:
    threshold_pct = st.slider("Linearity threshold (%)", 1, 50, 20,
        help="Max allowed curvature as % of peak curvature in range. Lower = stricter linear region.")
with c3:
    min_pts_tauc = st.number_input("Min points — Tauc fit", min_value=5, value=15, step=5)
with c4:
    min_pts_makula = st.number_input("Min points — Makula fit", min_value=5, value=15, step=5)
    
# INITIAL PREVIEW PLOT

st.markdown("---")
st.markdown("**Select the fitting ranges for both fits using the sliders below the graph.**")

fig0, ax0 = plt.subplots(figsize=(10, 6))
ax0.scatter(energy, modified_function, s=6, color='steelblue')
ax0.set_xlabel("Energy (eV)")
ax0.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
ax0.set_xticks(np.arange(1.0, 7.0, 0.25))
ax0.set_ylim([0, 10]); ax0.set_xlim([1.6, 6.2]); ax0.grid(True)
st.pyplot(fig0); plt.close(fig0)

# RANGE SLIDERS

e_min, e_max = float(energy.min()), float(energy.max())

tauc_fit_range = st.slider(
    "Range for Tauc fit  🟠",
    min_value=e_min, max_value=e_max,
    value=(e_min, e_max), step=0.01, key='tauc'
)
makula_fit_range = st.slider(
    "Range for Makula (y-offset) fit  🟢",
    min_value=e_min, max_value=e_max,
    value=(e_min, e_max), step=0.01, key='makula'
)

# Preview plot with range markers & curvature overlay
fig1, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})

ax_main, ax_curv = axes

# Main scatter
ax_main.scatter(energy, modified_function, s=6, color='steelblue', label='Data')
ax_main.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
ax_main.set_ylim([0, 10]); ax_main.set_xlim([1.6, 6.2]); ax_main.grid(True)
ax_main.text(1.72, 9.1,
             f'Tauc range:   {tauc_fit_range[0]:.2f} – {tauc_fit_range[1]:.2f} eV\n'
             f'Makula range: {makula_fit_range[0]:.2f} – {makula_fit_range[1]:.2f} eV',
             fontsize=8, family='monospace')
for x in tauc_fit_range:
    ax_main.axvline(x=x, color='orange', linestyle='--', linewidth=1)
for x in makula_fit_range:
    ax_main.axvline(x=x, color='green', linestyle='--', linewidth=1)

# Curvature subplot
ye_smooth = gaussian_filter1d(modified_function, sigma=sigma)
d2y_full  = np.gradient(np.gradient(ye_smooth, energy), energy)
ax_curv.plot(energy, np.abs(d2y_full), color='purple', linewidth=0.8, label='|d²y/dx²|')
ax_curv.set_ylabel("|d²y/dx²|", fontsize=8)
ax_curv.set_xlabel("Energy (eV)")
ax_curv.set_xlim([1.6, 6.2]); ax_curv.grid(True)
ax_curv.set_xticks(np.arange(1.0, 7.0, 0.5))
ax_curv.legend(fontsize=7)
for x in tauc_fit_range:
    ax_curv.axvline(x=x, color='orange', linestyle='--', linewidth=1)
for x in makula_fit_range:
    ax_curv.axvline(x=x, color='green', linestyle='--', linewidth=1)

st.pyplot(fig1); plt.close(fig1)

st.caption(
    "**Reading the curvature plot:** The linear region is where |d²y/dx²| is lowest "
    "(flattest). Set your slider range to enclose that region. Adjust the smoothing sigma "
    "if the curvature looks too noisy."
)

# CALCULATION

st.markdown("---")
if st.button("Start calculation"):

    # recompute in case transition changed
    modified_function = compute_modified(data['f(R)'].values, energy, transition)
    x_fit = np.linspace(1.6, 6.2, 100)

    # Tauc fit
    with st.spinner("Finding linear region for Tauc fit..."):
        result_tauc = method_derivative(
            energy, modified_function,
            tauc_fit_range[0], tauc_fit_range[1],
            int(min_pts_tauc), sigma, threshold_pct
        )

    if result_tauc is None:
        st.error("Tauc fit failed: no linear region found in the selected range. "
                 "Try widening the range, lowering the min points, or raising the threshold.")
        st.stop()

    slope_tauc, intercept_tauc, r2_tauc, size_tauc, x_best_tauc, y_best_tauc = result_tauc
    y_pred_tauc = slope_tauc * x_fit + intercept_tauc

    # Makula fit 
    with st.spinner("Finding linear region for Makula fit..."):
        result_m = method_derivative(
            energy, modified_function,
            makula_fit_range[0], makula_fit_range[1],
            int(min_pts_makula), sigma, threshold_pct
        )

    if result_m is None:
        st.error("Makula fit failed: no linear region found in the selected range. "
                 "Try widening the range, lowering the min points, or raising the threshold.")
        st.stop()

    slope_m, intercept_m, r2_m, size_m, x_best_m, y_best_m = result_m
    y_pred_m = slope_m * x_fit + intercept_m

    # Intersections
    intersection_tauc   = round(-intercept_tauc / slope_tauc, 3)
    intersection_makula = round((intercept_m - intercept_tauc) / (slope_tauc - slope_m), 3)

    # Results summary
    st.header("Results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Bandgap — Tauc x-axis intercept", f"{intersection_tauc} eV")
        st.write(f"R² = {r2_tauc:.6f} | points used = {size_tauc}")
        st.write(f"Slope = {slope_tauc:.4f} | Intercept = {intercept_tauc:.4f}")
    with res_col2:
        st.metric("Bandgap — Makula intersection", f"{intersection_makula} eV")
        st.write(f"R² = {r2_m:.6f} | points used = {size_m}")
        st.write(f"Slope = {slope_m:.4f} | Intercept = {intercept_m:.4f}")

    # plot
    st.subheader("Final Plot")
    final_fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(energy, modified_function, s=6, color='steelblue', label='Data')
    ax.scatter(x_best_tauc, y_best_tauc, color='orange', s=10, zorder=3,
               label=f'Tauc linear region ({size_tauc} pts)')
    ax.scatter(x_best_m, y_best_m, color='limegreen', s=10, zorder=3,
               label=f'Makula linear region ({size_m} pts)')
    ax.plot(x_fit, y_pred_tauc, color='red', linewidth=1.5,
            label=f'Tauc fit  R²={r2_tauc:.4f}  →  {intersection_tauc} eV')
    ax.plot(x_fit, y_pred_m, color='red', linewidth=1.5, linestyle='--',
            label=f'Makula fit  R²={r2_m:.4f}')
    ax.axvline(x=intersection_makula, color='green', linewidth=1.2,
               label=f'Makula intersection = {intersection_makula} eV')

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
    ax.set_xticks(np.arange(1.0, 7.0, 0.5))
    ax.set_ylim([0, 10]); ax.set_xlim([1.6, 6.2])
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=3, fancybox=True)
    st.pyplot(final_fig)

    # Downloads
    img_bytes = io.BytesIO()
    final_fig.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close(final_fig)

    fname = name_of_file if name_of_file else "uvvis_result"

    st.download_button("⬇ Download image (PNG)", data=img_bytes.getvalue(),
                       file_name=f'{fname}.png', mime="image/png")

    text_contents = f"""UV-Vis Analysis Results
=======================
Method: Derivative-based (curvature) — sigma={sigma}, threshold={threshold_pct}%

Bandgap (Tauc x-axis intercept): {intersection_tauc} eV
Bandgap (Makula intersection):   {intersection_makula} eV

Tauc fit:
  Range:     {tauc_fit_range[0]} – {tauc_fit_range[1]} eV
  R2:        {r2_tauc}
  Points:    {size_tauc}
  Slope:     {slope_tauc}
  Intercept: {intercept_tauc}

Makula fit:
  Range:     {makula_fit_range[0]} – {makula_fit_range[1]} eV
  R2:        {r2_m}
  Points:    {size_m}
  Slope:     {slope_m}
  Intercept: {intercept_m}
"""
    st.download_button("⬇ Download values (TXT)", data=text_contents,
                       file_name=f'{fname}.txt', mime="text/plain")
