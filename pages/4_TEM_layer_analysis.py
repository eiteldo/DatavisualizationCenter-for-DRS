import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
import streamlit as st


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return img


def extract_and_process_slice(img, y_pos, window_len, poly_order):
    horizontal_slice = img[y_pos, :]
    inverted_slice = np.max(horizontal_slice) - horizontal_slice
    return savgol_filter(inverted_slice, window_len, poly_order)


def auto_detect_peaks(smoothed_slice, prominence_factor=0.3):
    max_val = np.max(smoothed_slice)
    min_val = np.min(smoothed_slice)
    prominence = prominence_factor * (max_val - min_val)
    peaks, _ = find_peaks(smoothed_slice, prominence=prominence)
    return peaks, prominence


def evaluate_scan_line(img, y_pos, filter_window, filter_order, prominence_factor):
    try:
        sl = extract_and_process_slice(img, y_pos, filter_window, filter_order)
        peaks, prominence = auto_detect_peaks(sl, prominence_factor)
        if len(peaks) < 2:
            return 0, [], sl, prominence
        intensity_std = np.std(sl[peaks])
        spacing_std = np.std(np.diff(peaks))
        score = len(peaks) / (1 + intensity_std + spacing_std)
        return score, peaks, sl, prominence
    except Exception:
        return 0, [], np.array([]), 0


def find_optimal_scan_line(img, filter_window, filter_order, n_scan_lines, prominence_factor):
    height = img.shape[0]
    test_positions = np.linspace(height // 4, 3 * height // 4, n_scan_lines, dtype=int)

    best_score, best_y, best_peaks, best_slice, best_prom = 0, test_positions[0], [], np.array([]), 0
    results = []

    for y in test_positions:
        score, peaks, sl, prom = evaluate_scan_line(img, y, filter_window, filter_order, prominence_factor)
        results.append({"y_pos": int(y), "score": score, "num_peaks": len(peaks)})
        if score > best_score:
            best_score, best_y, best_peaks, best_slice, best_prom = score, y, peaks, sl, prom

    return int(best_y), best_peaks, best_slice, best_prom, results


def analyze_layer_periodicity(peaks, px_to_nm):
    if len(peaks) < 2:
        return {"avg_distance": 0, "std_distance": 0, "distances": [], "regularity_score": 0}
    distances_nm = np.diff(peaks) * px_to_nm
    avg = np.mean(distances_nm)
    std = np.std(distances_nm)
    reg = avg / (std + 1e-6)
    return {"avg_distance": avg, "std_distance": std, "distances": distances_nm, "regularity_score": reg}


def build_figure(img, optimal_y, smoothed_slice, peaks, scan_results, layer_analysis,
                 px_to_nm, figure_size=(15, 5)):
    sns.set_style("dark")
    fig = plt.figure(figsize=figure_size)

    y_positions = [r["y_pos"] for r in scan_results]
    scores = [r["score"] for r in scan_results]

    # Original image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img, cmap="gray")
    ax1.axhline(optimal_y, color="gold", linestyle="--", linewidth=2,
                label=f"Optimal line (y={optimal_y})")
    ax1.set_title("Original Image")
    ax1.legend(fontsize=8)
    ax1.grid(True, color="red", linestyle="--", alpha=0.3)

    # Profile and peaks
    ax2 = fig.add_subplot(1, 3, 2)
    x_range = range(len(smoothed_slice))
    ax2.plot(smoothed_slice, color="darkslategray", linewidth=2)
    ax2.fill_between(x_range, smoothed_slice, 0, color="lightyellow", alpha=0.5)
    ax2.scatter(peaks, smoothed_slice[peaks], marker="o", color="red", s=60, zorder=5)
    for i, p in enumerate(peaks):
        ax2.annotate(str(i + 1), (p, smoothed_slice[p]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax2.set_xlabel("X Position (px)")
    ax2.set_ylabel("Grayscale Value (a.u.)")
    ax2.set_title(f"Layer Profile ({len(peaks)} peaks)")
    ax2.grid(True, linestyle="--", color="gray", alpha=0.3)

    # Stats text
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    cv_pct = (layer_analysis["std_distance"] / layer_analysis["avg_distance"] * 100
              if layer_analysis["avg_distance"] > 0 else 0)
    stats = (
        f"AUTOMATED ANALYSIS RESULTS\n"
        f"{'='*28}\n\n"
        f"Optimal scan line : y = {optimal_y} px\n\n"
        f"Peak Detection\n"
        f"  Peaks found : {len(peaks)}\n"
        f"  Method      : Prominence-based\n\n"
        f"Layer Spacing\n"
        f"  Mean : {layer_analysis['avg_distance']:.2f} ± {layer_analysis['std_distance']:.2f} nm\n"
        f"  CV   : {cv_pct:.1f}%\n"
        f"Scale\n"
        f"  {px_to_nm:.4f} nm/px\n"
        f"  Scan lines tested : {len(scan_results)}"
    )
    ax3.text(0.05, 0.95, stats, transform=ax3.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    return fig


def run_tem_analysis():
    st.header("TEM Greyscale Profile Analyzer")

    # Sidebar settings
    with st.sidebar:
        st.header("Analysis Settings")

        st.subheader("Scale")
        scale_nm = st.number_input(
            "Scale bar length (nm)", value=64.29, min_value=0.01, step=0.01,
            help="Physical length represented by the scale bar")
        scale_px = st.number_input(
            "Scale bar length (px)", value=2048, min_value=1, step=1,
            help="Pixel length of the scale bar")

        st.subheader("Scan Lines")
        n_scan_lines = st.slider("Lines to test", 5, 50, 20)

        st.subheader("Smoothing")
        filter_window = st.slider("Savitzky-Golay window", 3, 51, 10, step=2)
        filter_order = st.slider("Savitzky-Golay order", 1, 5, 2)

        st.subheader("Peak Detection")
        prominence_factor = st.slider(
            "Prominence factor", 0.05, 0.8, 0.30, step=0.05,
            help="Fraction of dynamic range used as minimum peak prominence")

    px_to_nm = scale_nm / scale_px

    # File upload
    uploaded = st.file_uploader(
        "Upload TEM image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])

    if uploaded is None:
        st.info("Upload a TEM image to begin analysis.")
        return

    file_bytes = uploaded.read()

    # Run analysis
    with st.spinner("Running automated scan-line analysis…"):
        try:
            img = load_image_from_bytes(file_bytes)
        except ValueError as e:
            st.error(str(e))
            return

        st.caption(f"Image loaded: {img.shape[1]} × {img.shape[0]} px")

        optimal_y, peaks, smoothed_slice, prominence, scan_results = find_optimal_scan_line(
            img, filter_window, filter_order, n_scan_lines, prominence_factor)

        layer_analysis = analyze_layer_periodicity(peaks, px_to_nm)

        fig = build_figure(img, optimal_y, smoothed_slice, peaks,
                           scan_results, layer_analysis, px_to_nm)

    # Display
    st.pyplot(fig)
    plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimal scan line", f"y = {optimal_y} px")
    m2.metric("Peaks detected", len(peaks))
    m3.metric("Avg layer spacing",
              f"{layer_analysis['avg_distance']:.2f} nm" if layer_analysis["avg_distance"] else "—")

    if len(peaks) > 0:
        with st.expander("Peak Details", expanded=False):
            import pandas as pd
            rows = [
                {
                    "Peak #": i + 1,
                    "Position (px)": int(p),
                    "Position (nm)": round(p * px_to_nm, 2),
                    "Intensity (a.u.)": round(float(smoothed_slice[p]), 1),
                    "Spacing to next (nm)": (
                        round(float((peaks[i + 1] - p) * px_to_nm), 2)
                        if i + 1 < len(peaks) else "—"
                    ),
                }
                for i, p in enumerate(peaks)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Download
    buf = io.BytesIO()
    fig_dl = build_figure(img, optimal_y, smoothed_slice, peaks,
                          scan_results, layer_analysis, px_to_nm)
    fig_dl.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_dl)
    buf.seek(0)
    st.download_button(
        "Download analysis figure (PNG)",
        data=buf,
        file_name="tem_analysis.png",
        mime="image/png",
    )


if __name__ == "__main__":
    st.set_page_config(page_title="TEM Greyscale Analyzer", layout="wide")
    run_tem_analysis()
