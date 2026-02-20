import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
from scipy.signal import find_peaks, savgol_filter

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raman Peakfinder",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Raman Peakfinder & Plotter")

# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a two-column tab-separated Raman text file."""
    try:
        df = pd.read_csv(
            io.StringIO(file_bytes.decode("utf-8")),
            sep="\t",
            header=None,
            names=["Wavenumber", "Intensity"],
            comment="#",
        )
        df = df.dropna().astype(float)
        df = df.sort_values("Wavenumber").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Could not parse {filename}: {e}")
        return pd.DataFrame()


def als_baseline_correction(y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline estimation."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = lam * D.T @ D
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.diags(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def smooth(y: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    window = window if window % 2 == 1 else window + 1
    window = max(window, poly + 1)
    return savgol_filter(y, window_length=window, polyorder=poly)


def find_raman_peaks(
    x: np.ndarray,
    y: np.ndarray,
    height: float,
    prominence: float,
    min_width: int,
    distance: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    peaks, props = find_peaks(
        y,
        height=height,
        prominence=prominence,
        width=min_width,
        distance=distance,
    )
    return x[peaks], y[peaks], props


def build_figure(
    datasets: list[dict],
    xlim: tuple,
    show_peaks: bool,
    annotate_peaks: bool,
    font_size: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    for ds in datasets:
        x, y, label = ds["x"], ds["y"], ds["label"]
        color = ds.get("color")
        line, = ax.plot(x, y, label=label, color=color, linewidth=1.2)

        if show_peaks and "peak_x" in ds:
            px, py = ds["peak_x"], ds["peak_y"]
            ax.plot(px, py, "o", color=line.get_color(), markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            if annotate_peaks:
                for xi, yi in zip(px, py):
                    ax.annotate(
                        f"{xi:.0f}",
                        xy=(xi, yi),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=font_size - 2,
                        rotation=45,
                        color=line.get_color(),
                    )

    ax.set_xlabel(r"Raman Shift (cm$^{-1}$)", fontsize=font_size)
    ax.set_ylabel("Intensity (a.u.)", fontsize=font_size)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
    ax.tick_params(axis="x", labelsize=font_size - 1)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        fontsize=font_size - 1,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        frameon=False,
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


# ── sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Smoothing")
    apply_filter = st.checkbox("Apply Savitzky-Golay Filter", value=False)
    window_size = st.slider("Window Size (odd)", 3, 101, 21, step=2,
                            disabled=not apply_filter)
    poly_order = st.slider("Polynomial Order", 1, 5, 2, disabled=not apply_filter)

    st.subheader("Baseline")
    apply_baseline = st.checkbox("ALS Baseline Correction", value=False)
    if apply_baseline:
        als_lam = st.select_slider(
            "Smoothness (λ)", options=[1e3, 1e4, 1e5, 1e6, 1e7], value=1e5,
            format_func=lambda v: f"{v:.0e}"
        )
        als_p = st.slider("Asymmetry (p)", 0.001, 0.1, 0.01, step=0.001, format="%.3f")

    st.subheader("Peak Detection")
    show_peaks = st.checkbox("Show Peaks", value=True)
    annotate_peaks = st.checkbox("Annotate Peak Positions", value=True,
                                 disabled=not show_peaks)
    min_prominence = st.slider("Min Prominence (%)", 0, 50, 5,
        help="Peak must rise this % of the intensity range above its surroundings")
    min_width = st.slider("Min Peak Width (points)", 1, 20, 3)
    min_distance = st.slider("Min Peak Distance (points)", 1, 100, 10)

    st.subheader("Plot")
    x_min = st.number_input("X-axis min (cm⁻¹)", value=90, step=10)
    x_max = st.number_input("X-axis max (cm⁻¹)", value=2500, step=10)
    font_size = st.slider("Font Size", 8, 18, 12)
    plot_dpi = st.select_slider("Export DPI", [150, 300, 600, 1200], value=300)

# ── file upload ────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more Raman data files (.txt, two tab-separated columns)",
    type=["txt"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload a Raman .txt file to get started.")
    st.stop()

# ── process each file ──────────────────────────────────────────────────────────
datasets = []
all_peaks = []

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for idx, uf in enumerate(uploaded_files):
    df = load_file(uf.read(), uf.name)
    if df.empty:
        continue

    label = uf.name.rsplit(".", 1)[0]
    x = df["Wavenumber"].values
    y = df["Intensity"].values.copy()

    # baseline correction
    if apply_baseline:
        y = y - als_baseline_correction(y, lam=als_lam, p=als_p)

    # smoothing
    if apply_filter:
        y = smooth(y, window_size, poly_order)

    # normalise to [0, 1] per spectrum so sliders are universal
    y_range = y.max() - y.min() if y.max() != y.min() else 1.0
    y_norm = (y - y.min()) / y_range

    ds = {
        "label": label,
        "x": x,
        "y": y,
        "y_norm": y_norm,
        "color": COLOR_CYCLE[idx % len(COLOR_CYCLE)],
    }
    datasets.append(ds)

# ── per-file peak threshold (shown after loading) ──────────────────────────────
st.subheader("Peak Intensity Threshold (per file)")
threshold_cols = st.columns(min(len(datasets), 3))

for idx, ds in enumerate(datasets):
    col = threshold_cols[idx % len(threshold_cols)]
    with col:
        y = ds["y"]
        mn, mx = float(y.min()), float(y.max())
        thr = col.slider(
            ds["label"],
            min_value=mn,
            max_value=mx,
            value=mn + (mx - mn) * 0.3,
            step=(mx - mn) / 500 or 0.1,
            format="%.1f",
            key=f"thr_{idx}",
        )
        prom_abs = (mx - mn) * min_prominence / 100.0
        px, py, _ = find_raman_peaks(
            ds["x"], y,
            height=thr,
            prominence=prom_abs,
            min_width=min_width,
            distance=min_distance,
        )
        ds["peak_x"] = px
        ds["peak_y"] = py

        # collect for table
        for w, i in zip(px, py):
            all_peaks.append({"File": ds["label"], "Wavenumber (cm⁻¹)": round(w, 1), "Intensity": round(i, 2)})

# ── plot ───────────────────────────────────────────────────────────────────────
fig = build_figure(
    datasets,
    xlim=(x_min, x_max),
    show_peaks=show_peaks,
    annotate_peaks=annotate_peaks,
    font_size=font_size,
)

img_buf = io.BytesIO()
fig.savefig(img_buf, format="png", dpi=plot_dpi, bbox_inches="tight")
img_buf.seek(0)

st.pyplot(fig, use_container_width=True)

col1, col2 = st.columns(2)
col1.download_button(
    "⬇️ Download Plot (PNG)",
    data=img_buf,
    file_name="raman_plot.png",
    mime="image/png",
)

# ── peak table ─────────────────────────────────────────────────────────────────
if all_peaks:
    st.subheader("Detected Peaks")
    peak_df = pd.DataFrame(all_peaks).sort_values(["File", "Wavenumber (cm⁻¹)"])
    st.dataframe(peak_df, use_container_width=True, hide_index=True)

    csv_buf = peak_df.to_csv(index=False).encode()
    col2.download_button(
        "⬇️ Download Peaks (CSV)",
        data=csv_buf,
        file_name="raman_peaks.csv",
        mime="text/csv",
    )
