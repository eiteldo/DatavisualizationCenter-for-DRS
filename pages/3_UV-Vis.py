import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter1d

# page config
st.set_page_config(page_title="UV-Vis analysis tool", page_icon="📈", layout="wide")
st.title("📈 UV-Vis analysis tool")

# helper

def compute_modified(f_r, energy_arr, n):
    result = (f_r * energy_arr) ** n
    return np.where(np.isfinite(result), result, 0.0)


def fit_line(xb, yb):
    slope, intercept = np.polyfit(xb, yb, 1)
    ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
    ss_tot = np.sum((yb - yb.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def find_tauc_region(energy, mod, window_pts, sigma=2):
    """
    Finds the steep rising linear edge (Tauc fit).
    Score = R² × slope² — favours steep, linear windows.
    Ignores shallow Urbach tail.
    Fit done on original unsmoothed data.
    """
    mod_smooth = gaussian_filter1d(mod, sigma=sigma)
    n = len(energy)
    best_score = -np.inf
    best_idx = None

    for i in range(n - window_pts + 1):
        xb = energy[i : i + window_pts]
        yb = mod_smooth[i : i + window_pts]
        slope, intercept = np.polyfit(xb, yb, 1)
        if slope <= 0:
            continue
        ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
        ss_tot = np.sum((yb - yb.mean()) ** 2)
        if ss_tot == 0:
            continue
        r2 = 1.0 - ss_res / ss_tot
        score = r2 * (slope ** 2)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    xb = energy[best_idx : best_idx + window_pts]
    yb = mod[best_idx : best_idx + window_pts]
    slope, intercept, r2 = fit_line(xb, yb)
    return slope, intercept, r2, window_pts, xb.copy(), yb.copy()


def find_makula_region(energy, mod, tauc_start_energy, window_pts, sigma=2):
    # restrict search to the pre-onset region only
    pre_onset_mask = energy < tauc_start_energy
    xe = energy[pre_onset_mask]
    ye = mod[pre_onset_mask]
    n  = len(xe)

    if n < window_pts:
        return None

    ye_smooth = gaussian_filter1d(ye, sigma=sigma)
    eps = 1e-6  # prevents division by zero for perfectly flat segments

    best_score = -np.inf
    best_idx   = None

    for i in range(n - window_pts + 1):
        xb = xe[i : i + window_pts]
        yb = ye_smooth[i : i + window_pts]
        slope, intercept = np.polyfit(xb, yb, 1)
        ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
        ss_tot = np.sum((yb - yb.mean()) ** 2)
        if ss_tot == 0:
            continue
        r2    = 1.0 - ss_res / ss_tot
        # reward flatness: high R2, low slope
        score = r2 / (slope ** 2 + eps)
        if score > best_score:
            best_score = score
            best_idx   = i

    if best_idx is None:
        return None

    xb = xe[best_idx : best_idx + window_pts]
    yb = ye[best_idx : best_idx + window_pts]
    slope, intercept, r2 = fit_line(xb, yb)
    return slope, intercept, r2, window_pts, xb.copy(), yb.copy()

# file upload

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

# input

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    transition = st.number_input(
        "Type of transition",
        value=0.5, step=0.5,
        help="0.5 = indirect allowed, 2 = direct allowed"
    )
with c2:
    window_pts = st.slider(
        "Window width (data points)",
        min_value=10, max_value=150, value=30, step=5,
        help="Number of consecutive points used for each fitting window. "
             "Adjust to match the data of your instrument."
    )
with c3:
    name_of_file = st.text_input("File name for saving (without extension):")

modified_function = compute_modified(data['f(R)'].values, energy, transition)

# preview plot
st.markdown("---")
fig0, ax0 = plt.subplots(figsize=(10, 6))
ax0.scatter(energy, modified_function, s=6, color='steelblue')
ax0.set_xlabel("Energy (eV)")
ax0.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
ax0.set_xticks(np.arange(1.0, 7.0, 0.25))
ax0.set_ylim([0, 10]); ax0.set_xlim([1.6, 6.2]); ax0.grid(True)
st.pyplot(fig0); plt.close(fig0)

# calc

st.markdown("---")
if st.button("Start calculation"):

    modified_function = compute_modified(data['f(R)'].values, energy, transition)
    x_fit = np.linspace(1.6, 6.2, 100)

    # Tauc fit
    with st.spinner("Finding Tauc linear region..."):
        result_tauc = find_tauc_region(energy, modified_function, window_pts)

    if result_tauc is None:
        st.error("Tauc fit failed: no valid rising linear region found. "
                 "Try adjusting the window width.")
        st.stop()

    slope_tauc, intercept_tauc, r2_tauc, size_tauc, x_best_tauc, y_best_tauc = result_tauc
    y_pred_tauc = slope_tauc * x_fit + intercept_tauc
    tauc_start  = float(x_best_tauc[0])

    # Makula baseline fit
    with st.spinner("Finding Makula baseline (pre-onset region)..."):
        result_m = find_makula_region(
            energy, modified_function, tauc_start, window_pts
        )

    if result_m is None:
        st.error(
            f"Makula fit failed: not enough pre-onset data points below "
            f"{tauc_start:.3f} eV. Try reducing the window width."
        )
        st.stop()

    slope_m, intercept_m, r2_m, size_m, x_best_m, y_best_m = result_m
    y_pred_m = slope_m * x_fit + intercept_m

    # Intersections
    intersection_tauc   = round(-intercept_tauc / slope_tauc, 3)
    intersection_makula = round(
        (intercept_m - intercept_tauc) / (slope_tauc - slope_m), 3
    )

    # Results
    st.header("Results")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Bandgap — Tauc x-axis intercept", f"{intersection_tauc} eV")
        st.write(f"R² = {r2_tauc:.6f} | points = {size_tauc}")
        st.write(f"Slope = {slope_tauc:.4f} | Intercept = {intercept_tauc:.4f}")
        st.write(f"Region: {x_best_tauc[0]:.3f} – {x_best_tauc[-1]:.3f} eV")
    with res_col2:
        st.metric("Bandgap — Makula intersection", f"{intersection_makula} eV")
        st.write(f"R² = {r2_m:.6f} | points = {size_m}")
        st.write(f"Slope = {slope_m:.4f} | Intercept = {intercept_m:.4f}")
        st.write(f"Region: {x_best_m[0]:.3f} – {x_best_m[-1]:.3f} eV")

    # plot
    st.subheader("Final Plot")
    final_fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(energy, modified_function, s=6, color='steelblue', label='Data')
    ax.scatter(x_best_tauc, y_best_tauc, color='orange', s=12, zorder=3,
               label=f'Tauc region ({x_best_tauc[0]:.2f}–{x_best_tauc[-1]:.2f} eV)')
    ax.scatter(x_best_m, y_best_m, color='limegreen', s=12, zorder=3,
               label=f'Makula baseline ({x_best_m[0]:.2f}–{x_best_m[-1]:.2f} eV)')
    ax.plot(x_fit, y_pred_tauc, color='red', linewidth=1.5,
            label=f'Tauc fit =  {intersection_tauc} eV')
    ax.plot(x_fit, y_pred_m, color='darkgreen', linewidth=1.5, linestyle='--')
    ax.axvline(x=intersection_makula, color='green', linewidth=1.2,
               label=f'Makula fit = {intersection_makula} eV')

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
    ax.set_xticks(np.arange(1.0, 7.0, 0.5))
    ax.set_ylim([0, 10]); ax.set_xlim([1.6, 6.2])
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2, fancybox=True)
    st.pyplot(final_fig)

    # Downloads
    img_bytes = io.BytesIO()
    final_fig.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close(final_fig)

    fname = name_of_file if name_of_file else "uvvis_result"

    st.download_button("Download image (PNG)", data=img_bytes.getvalue(),
                       file_name=f'{fname}.png', mime="image/png")

    text_contents = f"""UV-Vis Analysis Results
=======================
Window width: {window_pts} points

Bandgap (Tauc x-axis intercept): {intersection_tauc} eV
Bandgap (Makula intersection):   {intersection_makula} eV

Tauc fit:
  Region:    {x_best_tauc[0]:.3f} - {x_best_tauc[-1]:.3f} eV
  R2:        {r2_tauc}
  Points:    {size_tauc}
  Slope:     {slope_tauc}
  Intercept: {intercept_tauc}

Makula baseline fit:
  Region:    {x_best_m[0]:.3f} - {x_best_m[-1]:.3f} eV
  R2:        {r2_m}
  Points:    {size_m}
  Slope:     {slope_m}
  Intercept: {intercept_m}
"""
    st.download_button("Download values (TXT)", data=text_contents,
                       file_name=f'{fname}.txt', mime="text/plain")

