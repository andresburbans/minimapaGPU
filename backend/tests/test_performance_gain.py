
import cupy as cp
import numpy as np
import time
import os
import sys
from pathlib import Path
import rasterio

# Add backend to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

import gpu_utils
gpu_utils.fix_cuda_dll_paths()
import render_gpu
from models import RenderConfig

def measure_fps():
    print("Measuring GPU Rendering FPS...")
    
    # Needs a real orthophoto to test full pipeline or a mock dataset
    # I'll use a dummy dataset structure if ortho not found, but let's try to find it.
    ortho_path = r"d:\Dev\MinimapaGPU\backend\data\ortho.tif"
    if not os.path.exists(ortho_path):
        # Fallback to current working dir or similar
        potential = list((ROOT.parent / "data").glob("*.tif"))
        if potential:
            ortho_path = str(potential[0])
        else:
            print("No ortho.tif found for realistic test. Using synthetic benchmark.")
            # Synthetic bench for kernels
            run_synthetic_bench()
            return

    with rasterio.open(ortho_path) as dataset:
        # Prepare context
        centers = [(dataset.bounds.left + 100, dataset.bounds.bottom + 100)]
        jobs = [(0, centers[0][0], centers[0][1], 45.0, "dummy.png")]
        
        config = RenderConfig(
            ortho_path=ortho_path,
            csv_path="",
            duration_sec=1.0,
            width=2048,
            height=2048,
            use_gpu=True
        )
        
        print("Preloading...")
        render_gpu.preload_track_gpu(config, jobs)
        
        # Warmup
        for _ in range(5):
            render_gpu.render_frame_gpu(dataset, [], centers[0][0], centers[0][1], 45.0, 2048, 2048, 150.0, 400, 60, 220, 0.18, 0.35, 120)

        # Benchmark FHD
        N = 50
        W_FHD, H_FHD = 1920, 1080
        start = time.time()
        for i in range(N):
            render_gpu.render_frame_gpu(dataset, [], centers[0][0], centers[0][1], 45.0 + i, W_FHD, H_FHD, 150.0, 400, 60, 220, 0.18, 0.35, 120)
            cp.cuda.Stream.null.synchronize()
            
        end = time.time()
        total = end - start
        fps = N / total
        print(f"Results for FHD (1920x1080):")
        print(f"Total time for {N} frames: {total:.4f}s")
        print(f"FPS: {fps:.2f}")
        with open("backend/tests/perf_result.txt", "w") as f:
            f.write(f"FPS: {fps:.2f}\nTotal: {total:.4f}\n")
        
        if fps > 25:
            print("🚀 EXCELLENT: GPU rendering is now fast enough for real-time visualization!")
        else:
            print("ℹ️ GPU rendering is improving but still bound by high resolution (2048x2048).")

def run_synthetic_bench():
    # Performance of individual kernels
    H, W = 2048, 2048
    data = cp.random.randint(0, 256, (4, 4000, 4000), dtype=cp.uint8)
    matrix = cp.eye(3, dtype=cp.float32)
    offset = cp.array([0, 100, 100], dtype=cp.float32)
    
    import cupyx.scipy.ndimage as ndimage
    N = 100
    start = time.time()
    for _ in range(N):
        ndimage.affine_transform(data, matrix, offset=offset, output_shape=(4, H, W))
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    print(f"Synthetic Kernel FPS (2K): {N/(end-start):.2f}")

if __name__ == "__main__":
    measure_fps()
