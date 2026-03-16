
import os
import sys
import numpy as np
import cupy as cp
import rasterio
from pathlib import Path
from PIL import Image

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import gpu_utils
import render_gpu

def test_consistency_and_perf():
    print("[TEST] Starting Consistency and Performance Verification...")
    
    if not render_gpu.HAS_GPU:
        print("[TEST] ERROR: GPU not available.")
        return

    VAL_DIR = Path("gpu_validation")
    ORTHO_PATH = VAL_DIR / "test_ortho_crop.tif"
    
    if not ORTHO_PATH.exists():
        print(f"[TEST] ERROR: {ORTHO_PATH} not found.")
        return

    W, H = 1920, 1080
    
    with rasterio.open(ORTHO_PATH) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        
        # 1. Preload
        render_gpu._CONTEXT.clear()
        render_gpu._CONTEXT.preload(ds, [(cx, cy)], 100.0)
        
        # 2. Render a frame with compass
        # We use a heading that should make the compass visible
        heading = 45.0
        frame_gpu = render_gpu.render_frame_gpu(
            None, [], cx, cy, heading, W, H, 50.0, 40, 60.0, 100, 0.3, 0.5, 20,
            show_compass=True, compass_size_px=50
        )
        
        # 3. CONSISTENCY CHECK: Ensure top-left (0,0,50,50) is empty 
        # (Assuming the background is empty there or just doesn't have a compass)
        # We check the first 50x50 pixels. If P1-A bug was present, a compass would be there.
        top_left = frame_gpu[0:50, 0:50, :].get()
        
        # A compass usually has many non-zero pixels.
        # Let's check how many non-zero pixels we have in top-left.
        # Note: If there's an ortho/background, there might be pixels, but 
        # we can compare it with a render WITHOUT compass.
        
        frame_no_compass = render_gpu.render_frame_gpu(
            None, [], cx, cy, heading, W, H, 50.0, 40, 60.0, 100, 0.3, 0.5, 20,
            show_compass=False
        )
        top_left_no = frame_no_compass[0:50, 0:50, :].get()
        
        diff = np.abs(top_left.astype(np.int16) - top_left_no.astype(np.int16))
        max_diff = np.max(diff)
        
        if max_diff > 0:
            print(f"[TEST] CONSISTENCY FAILED: Top-left corner differs by {max_diff} units between compass ON/OFF.")
            # Save for inspection
            Image.fromarray(top_left).save("test_top_left_fail.png")
        else:
            print("[TEST] CONSISTENCY PASSED: No ghost compass detected in top-left corner.")

        # 4. SPEED TEST
        print(f"[TEST] Running speed test (30 frames @ FHD)...")
        import time
        t0 = time.time()
        for i in range(30):
            _ = render_gpu.render_frame_gpu(
                None, [], cx, cy, i*10, W, H, 50.0, 40, 60.0, 100, 0.3, 0.5, 20,
                show_compass=True, compass_size_px=50
            )
        cp.cuda.Stream.null.synchronize()
        t1 = time.time()
        fps = 30 / (t1 - t0)
        print(f"[TEST] Current Performance: {fps:.2f} FPS")

if __name__ == "__main__":
    test_consistency_and_perf()
