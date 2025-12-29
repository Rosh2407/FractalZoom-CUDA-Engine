import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
import pandas as pd
from numba import cuda
import warnings

# Suppress Numba warnings
warnings.filterwarnings('ignore')

try:
    from fractal_backend import (
        IntelligentFractalGPU,
        OriginalFractalCPU,
        FractalVideoCompressor,
        FractalRenderer,
        VideoQualityAnalyzer
    )
except ImportError:
    st.error("Error: Could not import backend. Make sure 'fractal_backend.py' exists.")
    st.stop()

st.set_page_config(page_title="FractalZoom CUDA Engine", page_icon="🧊", layout="wide")

st.markdown("""
    <style>
    .stButton>button {width: 100%; background-color: #00CC96; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'final_stats' not in st.session_state:
    st.session_state['final_stats'] = []

with st.sidebar:
    st.title("⚙️ Engine Settings")
    mode = st.radio("Choose Operation:", [
        "Deep Zoom Render",
        "Benchmark (CPU vs GPU)",
        "Quality Validation (PSNR/SSIM)"
    ])
    st.markdown("---")

    if mode != "Quality Validation (PSNR/SSIM)":
        search_frac = st.slider("Search Fraction", 0.01, 0.50, 0.10)

    if mode == "Deep Zoom Render":
        selected_zooms = st.multiselect(
            "Select Zoom Levels to Process:",
            options=[1.5, 2.0, 3.0, 4.0, 8.0],
            default=[2.0, 3.0]
        )
        iterations = st.slider("Fractal Iterations", 1, 15, 10)

        # Add a Reset Button to clear the state if needed
        if st.button("🔄 Reset / New Upload"):
            st.session_state['processing_complete'] = False
            st.session_state['final_stats'] = []
            st.rerun()

    if cuda.is_available():
        st.success(f"✅ GPU: {cuda.get_current_device().name.decode('utf-8')}")

st.title("🧊 CUDA Fractal Video Interface (Colab Edition)")

# =========================================================
# MODE 1: DEEP ZOOM RENDER (MULTI-LEVEL + PERSISTENCE)
# =========================================================
if mode == "Deep Zoom Render":
    uploaded_file = st.file_uploader("Upload Source Video", type=["mp4", "avi"])

    if uploaded_file:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.info(f"Input: {width}x{height} | {total_frames} Frames")
        max_proc_frames = st.number_input("Max Frames", 10, total_frames, min(50, total_frames))

        # --- BUTTON LOGIC ---
        if st.button("✨ Compress & Zoom All"):
            if not selected_zooms:
                st.error("Please select at least one zoom level in the sidebar.")
            else:
                # Clear previous results
                st.session_state['final_stats'] = []

                compressor = FractalVideoCompressor()
                compressor.search_fraction = search_frac
                renderer = FractalRenderer()

                st.write("---")

                # --- PROCESSING LOOP ---
                for idx, current_zoom in enumerate(selected_zooms):
                    st.subheader(f"Processing Zoom Level: {current_zoom}x ({idx+1}/{len(selected_zooms)})")
                    progress_bar = st.progress(0)
                    frame_display = st.empty()

                    # Dimensions
                    aligned_w = (width // 16) * 16
                    aligned_h = (height // 16) * 16
                    out_w = int(aligned_w * current_zoom)
                    out_h = int(aligned_h * current_zoom)

                    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(temp_out.name, fourcc, fps, (out_w, out_h), True)

                    cap = cv2.VideoCapture(video_path)

                    start_time = time.time()

                    for i in range(max_proc_frames):
                        ret, frame = cap.read()
                        if not ret: break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        code = compressor.compress_frame(gray)
                        zoomed = renderer.render_frame(code, zoom_factor=current_zoom, iterations=iterations)

                        if zoomed.shape[0] != out_h or zoomed.shape[1] != out_w:
                            zoomed = cv2.resize(zoomed, (out_w, out_h))
                        zoomed_bgr = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2BGR)

                        writer.write(zoomed_bgr)

                        if i % 5 == 0:
                            progress_bar.progress((i + 1) / max_proc_frames)
                            frame_display.image(zoomed, caption=f"Zoom {current_zoom}x - Frame {i}", clamp=True, channels='GRAY')

                    cap.release()
                    writer.release()
                    progress_bar.progress(100)

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    file_size_mb = os.path.getsize(temp_out.name) / (1024 * 1024)
                    display_filename = f"fractal_{current_zoom}x.mp4"

                    # Store in SESSION STATE
                    st.session_state['final_stats'].append({
                        "filename": display_filename,
                        "resolution": f"{out_w}x{out_h}",
                        "size": file_size_mb,
                        "time": elapsed_time,
                        "path": temp_out.name
                    })

                # Mark processing as complete
                st.session_state['processing_complete'] = True

        # --- DISPLAY RESULTS (OUTSIDE THE BUTTON BLOCK) ---
        # This block runs every time the app refreshes, as long as 'processing_complete' is True
        if st.session_state['processing_complete'] and st.session_state['final_stats']:
            st.write("---")
            header = (
                "📊 OUTPUT FILE VERIFICATION & DECODING STATS\n"
                "================================================================================\n"
                f"{'Filename':<21}| {'Resolution':<13}| {'Size (MB)':<11}| {'Decode Time (s)':<16}|\n"
                "--------------------------------------------------------------------------------"
            )

            rows = ""
            for stat in st.session_state['final_stats']:
                rows += f"\n{stat['filename']:<21}| {stat['resolution']:<13}| {stat['size']:<11.2f}| {stat['time']:<16.2f}|"

            footer = "\n================================================================================"

            full_table = header + rows + footer
            st.code(full_table, language="text")

            st.subheader("💾 Download Results")
            # Create columns based on how many files we have
            cols = st.columns(len(st.session_state['final_stats']))

            for idx, stat in enumerate(st.session_state['final_stats']):
                with cols[idx]:
                    with open(stat['path'], "rb") as file:
                        st.download_button(
                            label=f"Download {stat['filename']}",
                            data=file,
                            file_name=stat['filename'],
                            mime="video/mp4",
                            key=f"dl_btn_{idx}"
                        )

# =========================================================
# MODE 2: BENCHMARK (Unchanged)
# =========================================================
elif mode == "Benchmark (CPU vs GPU)":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
        frames = []
        max_f = st.number_input("Frames to test", 10, 200, 20)

        for _ in range(max_f):
            ret, f = cap.read()
            if ret: frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cap.release()

        if st.button("🚀 Run Benchmark"):
            cpu_solver = OriginalFractalCPU()
            gpu_solver = IntelligentFractalGPU()
            gpu_solver.search_fraction = search_frac

            st.write("Running CPU...")
            res_cpu = cpu_solver.run(frames)
            st.write("Running GPU...")
            res_gpu = gpu_solver.run(frames)

            raw_mb = (frames[0].size * len(frames)) / (1024 * 1024)
            cpu_comp_mb = res_cpu['comp_size'] / (1024 * 1024)
            gpu_comp_mb = res_gpu['comp_size'] / (1024 * 1024)

            cpu_ratio = (raw_mb * 1024 * 1024) / res_cpu['comp_size'] if res_cpu['comp_size'] > 0 else 0
            gpu_ratio = (raw_mb * 1024 * 1024) / res_gpu['comp_size'] if res_gpu['comp_size'] > 0 else 0

            table_data = {
                "Metric": ["PSNR", "Time/Frame", "Ops/Frame", "Raw Size", "Comp Size", "Ratio"],
                "Original (CPU)": [f"{res_cpu['psnr']:.2f} dB", f"{res_cpu['time']:.2f} s", f"{int(res_cpu['ops']):,}", f"{raw_mb:.2f} MB", f"{cpu_comp_mb:.2f} MB", f"{cpu_ratio:.1f}:1"],
                "Intelligent (GPU)": [f"{res_gpu['psnr']:.2f} dB", f"{res_gpu['time']:.2f} s", f"{int(res_gpu['ops']):,}", f"{raw_mb:.2f} MB", f"{gpu_comp_mb:.2f} MB", f"{gpu_ratio:.1f}:1"]
            }
            st.subheader("🏆 Final Results Table")
            st.table(pd.DataFrame(table_data))

# =========================================================
# MODE 3: QUALITY VALIDATION (Unchanged)
# =========================================================
elif mode == "Quality Validation (PSNR/SSIM)":
    st.header("📊 Robust Video Quality Calculator")
    st.info("Compare the 'Original' video against the 'Fractal Zoomed' result.")

    col1, col2 = st.columns(2)
    with col1:
        f_ref = st.file_uploader("1. Upload Reference (Original)", type=["mp4"])
    with col2:
        f_dist = st.file_uploader("2. Upload Distorted (Fractal Output)", type=["mp4"])

    if f_ref and f_dist:
        t_ref = tempfile.NamedTemporaryFile(delete=False)
        t_ref.write(f_ref.read())

        t_dist = tempfile.NamedTemporaryFile(delete=False)
        t_dist.write(f_dist.read())

        max_chk_frames = st.number_input("Max Frames to Check", 10, 500, 50)

        if st.button("🔎 Analyze Quality"):
            analyzer = VideoQualityAnalyzer()

            with st.spinner("Calculating PSNR and SSIM..."):
                res = analyzer.compare_videos(t_ref.name, t_dist.name, max_frames=max_chk_frames)

            if "error" in res:
                st.error(res['error'])
            else:
                st.success(f"Analysis Complete on {res['frames']} frames")

                m1, m2 = st.columns(2)
                m1.metric("Average PSNR", f"{res['avg_psnr']:.2f} dB")
                m2.metric("Average SSIM", f"{res['avg_ssim']:.4f}")

                st.line_chart(pd.DataFrame({
                    "PSNR (dB)": res['psnr_history'],
                    "SSIM": res['ssim_history']
                }))
