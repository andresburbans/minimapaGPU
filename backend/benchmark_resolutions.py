
import os
import sys
import time
import numpy as np
import rasterio
from pathlib import Path
import cupy as cp

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import backend.gpu_utils as gpu_utils
import backend.render_gpu as render_gpu

def run_benchmark():
    if not render_gpu.HAS_GPU:
        print("[BENCHMARK] GPU not available.")
        return

    VAL_DIR = Path("gpu_validation")
    ORTHO_PATH = VAL_DIR / "test_ortho_crop.tif"
    
    if not ORTHO_PATH.exists():
        print(f"[BENCHMARK] {ORTHO_PATH} not found.")
        return

    # Helper to mock data
    def run_with_data(num_frames=30):
        with rasterio.open(ORTHO_PATH) as ds:
            # Generate mock centers
            bounds = ds.bounds
            cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
            radius = (bounds.right - bounds.left) * 0.1
            centers = []
            for i in range(num_frames):
                angle = 2 * np.pi * i / num_frames
                centers.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
            
            # Preload once within context
            print("[BENCHMARK] Preloading...")
            render_gpu._CONTEXT.clear()
            render_gpu._CONTEXT.preload(ds, centers, 150.0)
            cp.cuda.Stream.null.synchronize()
            
            # Resolutions to test
            resolutions = [
                ("nHD", 640, 360),
                ("FHD", 1920, 1080)
            ]
            
            for name, w, h in resolutions:
                print(f"\n[BENCHMARK] Testing {name} ({w}x{h})...")
                
                # Warmup
                for _ in range(5):
                     # signature: dataset, vectors, center_e, center_n, heading, width, height, map_half_width_m, arrow_size_px, cone_angle_deg, cone_length_px, cone_opacity, icon_circle_opacity, icon_circle_size_px
                     _ = render_gpu.render_frame_gpu(
                        None, [], centers[0][0], centers[0][1], 0, w, h, 
                        50.0, 40, 60.0, 100, 0.0, 0.5, 20, # Added missing args
                        show_compass=True, compass_size_px=50
                    )
                cp.cuda.Stream.null.synchronize()
                
                # Run
                t_start = time.time()
                for i in range(30):
                    ce, cn = centers[i]
                    heading = i * 10
                    _ = render_gpu.render_frame_gpu(
                        None, [], ce, cn, heading, w, h, 
                        50.0, 40, 60.0, 100, 0.0, 0.5, 20, # Added missing args
                        show_compass=True, compass_size_px=50
                    )
                cp.cuda.Stream.null.synchronize()
                t_end = time.time()
                
                fps = 30 / (t_end - t_start)
                print(f"[BENCHMARK] {name}: {fps:.2f} FPS (Time: {t_end - t_start:.2f}s)")

    run_with_data(30)

if __name__ == "__main__":
    run_benchmark()
