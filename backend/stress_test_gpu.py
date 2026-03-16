
import os
import sys
import time
import numpy as np
import rasterio
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import backend.gpu_utils as gpu_utils
import backend.render_gpu as render_gpu
from backend.render import load_vectors

def stress_test_preload():
    print("[STRESS] Starting GPU Preload Stress Test...")
    
    if not render_gpu.HAS_GPU:
        print("[STRESS] ERROR: GPU not available.")
        return

    VAL_DIR = Path("gpu_validation")
    ORTHO_PATH = VAL_DIR / "test_ortho_crop.tif"
    
    if not ORTHO_PATH.exists():
        print(f"[STRESS] ERROR: {ORTHO_PATH} not found.")
        return

    # FHD parameters
    W, H = 1920, 1080
    
    # Mock some data
    class MockConfig:
        ortho_path = str(ORTHO_PATH)
        vector_layers = []
        vectors_paths = [] # Empty for fastest cycles
        curves_path = None
        line_color = "red"
        line_width = 2
        boundary_color = "blue"
        boundary_width = 2
        point_color = "green"
        map_half_width_m = 50.0
        arrow_size_px = 40
        cone_length_px = 100
        wms_source = "google_hybrid"
        csv_path = "mock.csv"

    config = MockConfig()
    
    with rasterio.open(ORTHO_PATH) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        centers = [(cx, cy), (cx+10, cy+10)]
        
        # Load empty vectors
        vec_data = load_vectors(ds.crs, [], [], None, "red", 2, "blue", 2, "green")

        for i in range(20):
            print(f"\n--- Cycle {i+1}/20 ---")
            try:
                t0 = time.time()
                print("[STRESS] Clearing context...")
                render_gpu._CONTEXT.clear()
                
                print("[STRESS] Preloading...")
                render_gpu.preload_track_gpu(config, [(0, cx, cy, 0)])
                
                print("[STRESS] Rendering one frame...")
                _ = render_gpu.render_frame_gpu(
                    ds, [], cx, cy, 0.0, W, H, 50.0, 40, 60.0, 100, 0.3, 0.5, 20
                )
                
                t1 = time.time()
                print(f"[STRESS] Cycle {i+1} finished in {t1-t0:.2f}s")
                
            except Exception as e:
                print(f"[STRESS] !!! FAILED at cycle {i+1}: {e}")
                # Print stack trace
                import traceback
                traceback.print_exc()
                break

if __name__ == "__main__":
    stress_test_preload()
