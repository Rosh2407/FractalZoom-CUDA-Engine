# FractalZoom CUDA Engine 🧊

An intelligent, GPU-accelerated video compression and deep-zoom rendering engine. This project leverages **NVIDIA CUDA** via Numba to perform real-time fractal analysis and reconstruction, allowing for extreme upscaling through self-similarity.

## 🚀 Overview

Fractal compression is based on the **Partitioned Iterated Function System (PIFS)**. Unlike traditional compression (like JPEG), fractal compression stores the "rules" to recreate an image rather than the pixels themselves. This project optimizes that process using a custom CUDA kernel to achieve massive speedups over traditional CPU-based fractal methods.

### Key Features
* **CUDA-Accelerated Matching:** Custom Numba kernels parallelize the search for the best Domain-to-Range block matches.
* **Intelligent Search:** Uses variance-based sorting and search-window pruning to reduce computational complexity.
* **Deep Zoom Rendering:** Reconstructs video at 2x, 4x, or 8x zoom levels by iteratively applying fractal transforms.
* **Quality Analytics Suite:** Built-in tools to calculate **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) to validate output quality.
* **Interactive UI:** Streamlit-based dashboard for easy video uploading, parameter tuning, and benchmarking.

---

## 🛠️ Technical Deep Dive

### The CUDA Smart Match Kernel
The core bottleneck of fractal compression is the "Domain Search." Our implementation uses a `smart_match_kernel` to solve this in parallel:

1.  **Thread Mapping:** Each CUDA thread processes one Range block.
2.  **Least-Squares Regression:** The kernel calculates the optimal contrast scaling ($s$) and brightness offset ($o$) for every block simultaneously.
3.  **Early Exit Optimization:** Threads "break" early if a match exceeds a quality threshold, saving GPU cycles.
4.  **Hardware Efficiency:** Uses `float32` precision and 1D/2D grid mapping optimized for the **T4 GPU**.

### The Rendering Process
To achieve a "Deep Zoom," the `FractalRenderer` applies the contractive mapping principle. It takes the compressed fractal codes and iteratively renders them onto an upscaled canvas (default 10 iterations), filling in high-frequency details that traditional interpolation would miss.

---

## 📊 Performance Benchmark
The engine includes a comparison mode between:
* **Original CPU:** Single-threaded Python implementation using NumPy.
* **Intelligent GPU:** Parallelized CUDA implementation with search pruning.

| Metric | Original (CPU) | Intelligent (GPU) |
| :--- | :--- | :--- |
| **Speed** | 1.0x (Baseline) | ~20x - 50x Faster |
| **Search Logic** | Full Exhaustive | Variance-Pruned (Smart) |
| **Throughput** | Sequential | Massively Parallel |

---

## 💻 Installation & Setup

### Prerequisites
* NVIDIA GPU (e.g., Tesla T4, RTX series)
* Python 3.10+
* CUDA Toolkit
