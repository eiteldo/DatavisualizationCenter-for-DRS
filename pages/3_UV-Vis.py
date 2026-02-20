import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter1d

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UV-Vis analysis tool", page_icon="📈", layout="wide")
st.title("📈 UV-Vis analysis tool")

# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def compute_modified(f_r, energy_arr, n):
    result = (f_r * energy_arr) ** n
    return np.where(np.isfinite(result), result, 0.0)


def find_linear_region(energy, mod, window_pts, sigma=2):
    """
    Automatically finds the most prominent linear region in a Tauc/Makula plot.

    Scoring = R² × slope² × window_pts
      - R²        → penalises non-linear windows
      - slope²    → favours steep rising edges, ignores shallow Urbach tail
                    and flat baseline; squared so sign doesn't matter but
                    large slopes dominate
      - window_pts → all windows are the same width so this is constant,
                     but kept explicit for clarity

    Steps:
      1. Light Gaussian smoothing to reduce noise before scoring.
      2. Slide a fixed-width window across the full energy axis.
      3. Score every window.
      4. Pick the highest-scoring window.
      5. Fit the line on the ORIGINAL (unsmoothed) data at that position.

    Returns (slope, intercept, r2, window_pts, x_subset, y_subset) or None.
    """
    n = len(energy)
    if n < window_pts:
        return None

    mod_smooth = gaussian_filter1d(mod, sigma=sigma)

    best_score = -np.inf
    best_idx = None

    for i in range(n - window_pts + 1):
        xb = energy[i : i + window_pts]
        yb = mod_smooth[i : i + window_pts]

        slope, intercept = np.polyfit(xb, yb, 1)

        # only consider windows with a positive slope (rising edge)
        if slope <= 0:
            continue

        ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
        ss_tot = np.sum((yb - yb.mean()) ** 2)
        if ss_tot == 0:
            continue

        r2 = 1.0 - ss_res / ss_tot
        score = r2 * (slope ** 2) * window_pts

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    # fit on original unsmoothed data
    xb = energy[best_idx : best_idx + window_pts]
    yb = mod[best_idx : best_idx + window_pts]
    slope, intercept = np.polyfit(xb, yb, 1)
    ss_res = np.sum((yb - (slope * xb + intercept)) ** 2)
    ss_tot = np.sum((yb - yb.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return slope, intercept, r2, window_pts, xb.copy(), yb.copy()


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD & DATA
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────

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
        help="Number of consecutive data points used for fitting. "
             "Increase for instruments with higher data density, "
             "decrease if spectra are short."
    )
with c3:
    name_of_file = st.text_input("File name for saving (without extension):")

modified_function = compute_modified(data['f(R)'].values, energy, transition)

# ── Preview plot ──────────────────────────────────────────────────────────────
st.markdown("---")
fig0, ax0 = plt.subplots(figsize=(10, 6))
ax0.scatter(energy, modified_function, s=6, color='steelblue')
ax0.set_xlabel("Energy (eV)")
ax0.set_ylabel(fr"$(f(R)\cdot h\nu)^{{{transition}}}$")
ax0.set_xticks(np.arange(1.0, 7.0, 0.25))
ax0.set_ylim([0, 10]); ax0.set_xlim([1.6, 6.2]); ax0.grid(True)
st.pyplot(fig0); plt.close(fig0)

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
if st.button("▶  Start calculation"):

    modified_function = compute_modified(data['f(R)'].values, energy, transition)
    x_fit = np.linspace(1.6, 6.2, 100)

    # ── Tauc fit ──────────────────────────────────────────────────────────────
    with st.spinner("Finding linear region for Tauc fit..."):
        result_tauc = find_linear_region(energy, modified_function, window_pts)

    if result_tauc is None:
        st.error("Tauc fit failed: no valid rising linear region found. "
                 "Try adjusting the window width.")
        st.stop()

    slope_tauc, intercept_tauc, r2_tauc, size_tauc, x_best_tauc, y_best_tauc = result_tauc
    y_pred_tauc = slope_tauc * x_fit + intercept_tauc

    # ── Makula fit ────────────────────────────────────────────────────────────
    # Makula baseline sits below the Tauc linear region — score without slope²
    # so it can find shallower but well-fitting segments outside the Tauc region
    with st.spinner("Finding linear region for Makula fit..."):
        # exclude the Tauc region so Makula finds a different segment
        tauc_start_e = x_best_tauc[0]
        tauc_end_e   = x_best_tauc[-1]

        # mask out the Tauc region from the search
        energy_m = energy.copy()
        mod_m    = modified_function.copy()
        exclude  = (energy_m >= tauc_start_e) & (energy_m <= tauc_end_e)
        # set excluded region to NaN so the sliding window avoids it
        mod_m_search = mod_m.copy().astype(float)
        mod_m_search[exclude] = np.nan

        # sliding window for Makula — score by R² only (shallower baseline region)
        n = len(energy_m)
        best_score_m = -np.inf
        best_idx_m   = None
        mod_smooth_m = gaussian_filter1d(
            np.where(np.isfinite(mod_m_search), mod_m_search, 0.0), sigma=2
        )

        for i in range(n - window_pts + 1):
            xb = energy_m[i : i + window_pts]
            yb_orig = mod_m_search[i : i + window_pts]
            yb      = mod_smooth_m[i : i + window_pts]

            # skip windows that overlap the excluded Tauc region
            if np.any(np.isnan(yb_orig)):
                continue

            slope_w, intercept_w = np.polyfit(xb, yb, 1)
            ss_res = np.sum((yb - (slope_w * xb + intercept_w)) ** 2)
            ss_tot = np.sum((yb - yb.mean()) ** 2)
            if ss_tot == 0:
                continue

            r2_w  = 1.0 - ss_res / ss_tot
            # score: R² × length, no slope penalty so baseline is found too
            score = r2_w * window_pts

            if score > best_score_m:
                best_score_m = score
                best_idx_m   = i

    if best_idx_m is None:
        st.error("Makula fit failed: no valid linear baseline region found outside "
                 "the Tauc region. Try adjusting the window width.")
        st.stop()

    xb_m = energy_m[best_idx_m : best_idx_m + window_pts]
    yb_m = mod_m[best_idx_m : best_idx_m + window_pts]
    slope_m, intercept_m = np.polyfit(xb_m, yb_m, 1)
    ss_res_m = np.sum((yb_m - (slope_m * xb_m + intercept_m)) ** 2)
    ss_tot_m = np.sum((yb_m - yb_m.mean()) ** 2)
    r2_m   = 1.0 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0.0
    size_m = window_pts
    x_best_m, y_best_m = xb_m.copy(), yb_m.copy()
    y_pred_m = slope_m * x_fit + intercept_m

    # ── Intersections ─────────────────────────────────────────────────────────
    intersection_tauc   = round(-intercept_tauc / slope_tauc, 3)
    intersection_makula = round(
        (intercept_m - intercept_tauc) / (slope_tauc - slope_m), 3
    )

    # ── Results ───────────────────────────────────────────────────────────────
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

    # ── Final plot ────────────────────────────────────────────────────────────
    st.subheader("Final Plot")
    final_fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(energy, modified_function, s=6, color='steelblue', label='Data')
    ax.scatter(x_best_tauc, y_best_tauc, color='orange', s=12, zorder=3,
               label=f'Tauc linear region ({size_tauc} pts, '
                     f'{x_best_tauc[0]:.2f}–{x_best_tauc[-1]:.2f} eV)')
    ax.scatter(x_best_m, y_best_m, color='limegreen', s=12, zorder=3,
               label=f'Makula linear region ({size_m} pts, '
                     f'{x_best_m[0]:.2f}–{x_best_m[-1]:.2f} eV)')
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
              mode="expand", borderaxespad=0, ncol=2, fancybox=True)
    st.pyplot(final_fig)

    # ── Downloads ─────────────────────────────────────────────────────────────
    img_bytes = io.BytesIO()
    final_fig.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close(final_fig)

    fname = name_of_file if name_of_file else "uvvis_result"

    st.download_button("⬇ Download image (PNG)", data=img_bytes.getvalue(),
                       file_name=f'{fname}.png', mime="image/png")

    text_contents = f"""UV-Vis Analysis Results
=======================
Window width: {window_pts} points

Bandgap (Tauc x-axis intercept): {intersection_tauc} eV
Bandgap (Makula intersection):   {intersection_makula} eV

Tauc fit:
  Region:    {x_best_tauc[0]:.3f} – {x_best_tauc[-1]:.3f} eV
  R2:        {r2_tauc}
  Points:    {size_tauc}
  Slope:     {slope_tauc}
  Intercept: {intercept_tauc}

Makula fit:
  Region:    {x_best_m[0]:.3f} – {x_best_m[-1]:.3f} eV
  R2:        {r2_m}
  Points:    {size_m}
  Slope:     {slope_m}
  Intercept: {intercept_m}
"""
    st.download_button("Download values (TXT)", data=text_contents,
                       file_name=f'{fname}.txt', mime="text/plain")
