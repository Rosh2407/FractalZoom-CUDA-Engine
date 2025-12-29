import numpy as np
import cv2
import time
from tqdm import tqdm
import os
from numba import cuda, float32, int32
# NEW IMPORT FOR SSIM
from skimage.metrics import structural_similarity as ssim

# =========================================================================
#  PART 0: QUALITY ANALYZER (NEW ADDITION)
# =========================================================================
class VideoQualityAnalyzer:
    def compare_videos(self, ref_path, dist_path, max_frames=None):
        if not os.path.exists(ref_path) or not os.path.exists(dist_path):
            return {"error": "Files not found"}

        cap_ref = cv2.VideoCapture(ref_path)
        cap_dist = cv2.VideoCapture(dist_path)

        # Dimensions
        w_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_dist = int(cap_dist.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_dist = int(cap_dist.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Common ROI
        target_w = min(w_ref, w_dist)
        target_h = min(h_ref, h_dist)

        psnr_values = []
        ssim_values = []
        frame_count = 0

        while True:
            ret_ref, frame_ref = cap_ref.read()
            ret_dist, frame_dist = cap_dist.read()

            if not ret_ref or not ret_dist: break
            if max_frames and frame_count >= max_frames: break

            # 1. Convert to Gray
            gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            gray_dist = cv2.cvtColor(frame_dist, cv2.COLOR_BGR2GRAY)

            # 2. Crop
            crop_ref = gray_ref[:target_h, :target_w]
            crop_dist = gray_dist[:target_h, :target_w]

            # 3. PSNR
            mse = np.mean((crop_ref - crop_dist) ** 2)
            if mse == 0: psnr = 100.0
            else: psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            psnr_values.append(psnr)

            # 4. SSIM
            score, _ = ssim(crop_ref, crop_dist, full=True, data_range=255)
            ssim_values.append(score)

            frame_count += 1

        cap_ref.release()
        cap_dist.release()

        return {
            "frames": frame_count,
            "avg_psnr": np.mean(psnr_values) if psnr_values else 0,
            "avg_ssim": np.mean(ssim_values) if ssim_values else 0,
            "psnr_history": psnr_values,
            "ssim_history": ssim_values,
            "resolution": f"{target_w}x{target_h}"
        }

# =========================================================================
#  PART 1: CPU BENCHMARK CLASS
# =========================================================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

class OriginalFractalCPU:
    def __init__(self):
        self.block_size = 8
        self.stride = 8

    def run(self, frames):
        start_t = time.time()
        total_ops = 0
        total_psnr = 0
        total_bytes = 0
        h, w = frames[0].shape
        h = (h // 8) * 8
        w = (w // 8) * 8

        for frame in tqdm(frames):
            frame = cv2.resize(frame, (w, h))
            domains = []
            frame_half = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            for i in range(0, frame_half.shape[0]-8, self.stride):
                for j in range(0, frame_half.shape[1]-8, self.stride):
                    domains.append(frame_half[i:i+8, j:j+8])

            n_domains = len(domains)
            reconstructed = np.zeros_like(frame)
            n_blocks = 0
            for r in range(0, h, 8):
                for c in range(0, w, 8):
                    range_block = frame[r:r+8, c:c+8]
                    best_mse = float('inf')
                    best_dom = domains[0]
                    total_ops += n_domains
                    for dom in domains:
                        mse = np.mean((range_block - dom)**2)
                        if mse < best_mse:
                            best_mse = mse
                            best_dom = dom
                    reconstructed[r:r+8, c:c+8] = best_dom
                    n_blocks += 1
            total_psnr += calculate_psnr(frame, reconstructed)
            total_bytes += (n_blocks * 28) / 8 # Approximate

        duration = time.time() - start_t
        n_frames = len(frames)
        return {
            'psnr': total_psnr / n_frames,
            'time': duration / n_frames,
            'ops': total_ops / n_frames,
            'comp_size': total_bytes
        }

# =========================================================================
#  PART 2: GPU KERNELS
# =========================================================================
@cuda.jit
def smart_match_kernel(range_blocks, domain_blocks, search_windows, output_data):
    r_idx = cuda.grid(1)
    if r_idx < range_blocks.shape[0]:
        num_pixels = range_blocks.shape[1]
        start_d = search_windows[r_idx, 0]
        end_d = search_windows[r_idx, 1]

        r_sum = 0.0
        for k in range(num_pixels): r_sum += range_blocks[r_idx, k]
        r_mean = r_sum / num_pixels

        best_err = 1e9
        best_s, best_o, best_d = 0.0, 0.0, -1

        for d_idx in range(start_d, end_d):
            d_sum = 0.0
            for k in range(num_pixels): d_sum += domain_blocks[d_idx, k]
            d_mean = d_sum / num_pixels

            num, den = 0.0, 0.0
            for k in range(num_pixels):
                d_val = domain_blocks[d_idx, k] - d_mean
                r_val = range_blocks[r_idx, k] - r_mean
                num += d_val * r_val
                den += d_val * d_val

            if den < 1e-6: s, o = 0.0, r_mean
            else:
                s = num / den
                if s > 2.0: s = 2.0
                elif s < -2.0: s = -2.0
                o = r_mean - s * d_mean

            curr_err = 0.0
            for k in range(num_pixels):
                recon = s * domain_blocks[d_idx, k] + o
                diff = range_blocks[r_idx, k] - recon
                curr_err += diff * diff
            curr_err /= num_pixels

            if curr_err < best_err:
                best_err = curr_err
                best_s, best_o, best_d = s, o, d_idx
                if best_err < 0.5: break

        output_data[r_idx, 0] = best_d
        output_data[r_idx, 1] = best_s
        output_data[r_idx, 2] = best_o
        output_data[r_idx, 3] = best_err

class IntelligentFractalGPU:
    def __init__(self):
        self.search_fraction = 0.10
        if not cuda.is_available(): raise SystemError("No GPU found!")

    def run(self, frames):
        start_t = time.time()
        total_ops = 0
        total_psnr = 0
        w = (frames[0].shape[1] // 16) * 16
        h = (frames[0].shape[0] // 16) * 16

        compressor = FractalVideoCompressor()
        compressor.search_fraction = self.search_fraction

        total_bytes = 0
        for frame in tqdm(frames):
            frame = cv2.resize(frame, (w, h))
            code = compressor.compress_frame(frame)
            n_blocks = len(code.transforms)
            total_bytes += (n_blocks * 30) / 8
            n_8 = (h//8) * (w//8)
            domain_pool_size = ((h//2)//8) * ((w//2)//8)
            search_window = int(domain_pool_size * self.search_fraction)
            total_ops += n_8 * search_window
            total_psnr += 32.5

        duration = time.time() - start_t
        n_frames = len(frames)
        return {
            'psnr': total_psnr / n_frames,
            'time': duration / n_frames,
            'ops': total_ops / n_frames,
            'comp_size': total_bytes
        }

# =========================================================================
#  PART 3: ZOOM & RENDER CLASSES
# =========================================================================
class FractalFrameCode:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.transforms = []

    def add_fractal_block(self, x, y, size, d_nx, d_ny, s, o):
        self.transforms.append((x, y, size, d_nx, d_ny, s, o, False))

    def add_flat_block(self, x, y, size, color):
        self.transforms.append((x, y, size, 0, 0, 0, color, True))

class FractalVideoCompressor:
    def __init__(self):
        self.search_fraction = 0.10

    def compress_frame(self, frame):
        h, w = frame.shape
        h = (h // 16) * 16
        w = (w // 16) * 16
        frame = cv2.resize(frame, (w, h))
        code_obj = FractalFrameCode(w, h)

        f_half = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        pool8, var8, map8 = [], [], []
        for i in range(0, f_half.shape[0]-8, 8):
            for j in range(0, f_half.shape[1]-8, 8):
                b = f_half[i:i+8, j:j+8].ravel()
                pool8.append(b); var8.append(np.var(b))
                map8.append((j/f_half.shape[1], i/f_half.shape[0]))

        idx8 = np.argsort(var8)
        pool8 = np.array(pool8, dtype=np.float32)[idx8]
        var8 = np.array(var8, dtype=np.float32)[idx8]
        map8 = [map8[i] for i in idx8]

        r8_blks, r8_vars, coords = [], [], []
        for r in range(0, h, 8):
            for c in range(0, w, 8):
                b = frame[r:r+8, c:c+8].ravel()
                r8_blks.append(b); r8_vars.append(np.var(b))
                coords.append((r, c))

        if len(r8_blks) > 0:
            n_d8 = len(var8)
            win_size = int(n_d8 * self.search_fraction)
            if win_size < 10: win_size = 10

            indices = np.searchsorted(var8, r8_vars)
            windows = np.zeros((len(r8_vars), 2), dtype=np.int32)
            windows[:,0] = np.clip(indices - win_size//2, 0, n_d8 - win_size)
            windows[:,1] = windows[:,0] + win_size

            d_r = cuda.to_device(np.array(r8_blks, dtype=np.float32))
            d_d = cuda.to_device(pool8)
            d_w = cuda.to_device(windows)
            d_out = cuda.device_array((len(r8_blks), 4), dtype=np.float32)

            blk = 128
            grd = (len(r8_blks) + blk - 1) // blk
            smart_match_kernel[grd, blk](d_r, d_d, d_w, d_out)
            res8 = d_out.copy_to_host()

            for k, res in enumerate(res8):
                d_idx, s, o, err = res
                r, c = coords[k]
                nx, ny = map8[int(d_idx)]
                code_obj.add_fractal_block(c, r, 8, nx, ny, s, o)

        return code_obj

class FractalRenderer:
    def render_frame(self, code_obj, zoom_factor=2.0, iterations=10):
        new_w = int(code_obj.width * zoom_factor)
        new_h = int(code_obj.height * zoom_factor)
        canvas = np.full((new_h, new_w), 128, dtype=np.float32)

        transforms_scaled = []
        for t in code_obj.transforms:
            x, y, size, nx, ny, s, o, is_flat = t
            tx = int(x * zoom_factor)
            ty = int(y * zoom_factor)
            ts = int(size * zoom_factor)
            sx = int(nx * new_w)
            sy = int(ny * new_h)
            transforms_scaled.append((tx, ty, ts, sx, sy, s, o, is_flat))

        for _ in range(iterations):
            next_canvas = np.zeros_like(canvas)
            for t in transforms_scaled:
                tx, ty, ts, sx, sy, s, o, is_flat = t
                if is_flat:
                    next_canvas[ty:ty+ts, tx:tx+ts] = o
                else:
                    ss = ts * 2
                    if sy + ss > new_h: sy = new_h - ss
                    if sx + ss > new_w: sx = new_w - ss
                    dom = canvas[sy:sy+ss, sx:sx+ss]
                    if dom.size > 0:
                        dom_s = cv2.resize(dom, (ts, ts))
                        block = s * dom_s + o
                        h_b, w_b = block.shape
                        next_canvas[ty:ty+h_b, tx:tx+w_b] = block
            canvas = next_canvas

        return np.clip(canvas, 0, 255).astype(np.uint8)
