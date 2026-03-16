"""
Test that simulates the FULL app.py flow INCLUDING pinned memory allocation.
This is the critical difference - the web app allocates pinned memory for FFmpeg.
"""
import os
import sys
import time
import gc

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_utils
import numpy as np

GPU_RENDER_AVAILABLE = False
try:
    import cupy as cp
    from render_gpu import (
        init_gpu,
        render_frame_gpu,
        preload_track_gpu,
        cleanup_gpu,
        _CONTEXT,
        HAS_GPU
    )
    GPU_RENDER_AVAILABLE = HAS_GPU
    print(f"[TEST] GPU_RENDER_AVAILABLE = {GPU_RENDER_AVAILABLE}")
except ImportError as e:
    print(f"[TEST] GPU import failed: {e}")

import rasterio
from pathlib import Path

# Global to simulate server state
PINNED_MEM_GLOBAL = None
PINNED_BUFFER_GLOBAL = None

def allocate_pinned_memory(width, height):
    """Simulates what app.py does before preload"""
    global PINNED_MEM_GLOBAL, PINNED_BUFFER_GLOBAL
    
    alloc_size = width * height * 4
    print(f"[TEST] Allocating pinned memory: {alloc_size} bytes...")
    
    try:
        PINNED_MEM_GLOBAL = cp.cuda.alloc_pinned_memory(alloc_size)
        PINNED_BUFFER_GLOBAL = np.frombuffer(PINNED_MEM_GLOBAL, np.uint8)[:alloc_size].reshape((height, width, 4))
        print("[TEST] Pinned memory allocated successfully.")
        return True
    except Exception as e:
        print(f"[TEST] Pinned memory allocation failed: {e}")
        return False

def free_pinned_memory():
    """Free pinned memory"""
    global PINNED_MEM_GLOBAL, PINNED_BUFFER_GLOBAL
    PINNED_BUFFER_GLOBAL = None
    PINNED_MEM_GLOBAL = None
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    print("[TEST] Pinned memory freed.")

def simulate_full_render_task():
    """Simulates the FULL _render_task flow including pinned memory"""
    
    ORTHO_PATH = Path("gpu_validation") / "test_ortho_crop.tif"
    GEOJSON_PATH = r"G:\VIDEO-RIVERA\Shapes\Geojsons\FincasBEP.json"
    
    if not ORTHO_PATH.exists():
        print(f"[TEST] ERROR: {ORTHO_PATH} not found.")
        return False
    
    class MockConfig:
        ortho_path = str(ORTHO_PATH)
        csv_path = "mock.csv"
        vector_layers = []
        vectors_paths = [GEOJSON_PATH] if os.path.exists(GEOJSON_PATH) else []
        curves_path = None
        line_color = "#FF0000"
        line_width = 2
        boundary_color = "#0000FF"
        boundary_width = 2
        point_color = "#00FF00"
        width = 1920
        height = 1080
        map_half_width_m = 100.0
        arrow_size_px = 40
        cone_angle_deg = 60.0
        cone_length_px = 100
        cone_opacity = 0.3
        icon_circle_opacity = 0.5
        icon_circle_size_px = 20
        show_compass = True
        compass_size_px = 50
        fps = 30
        use_gpu = True
        wms_source = "google_hybrid"

    config = MockConfig()
    
    # STEP 1: Allocate pinned memory FIRST (like app.py does)
    print("[TEST] STEP 1: Allocating pinned memory (simulating FFmpeg prep)...")
    if not allocate_pinned_memory(config.width, config.height):
        return False
    
    # STEP 2: Now try to preload (this is where the error occurs!)
    print("[TEST] STEP 2: Calling preload_track_gpu (THIS IS WHERE ERROR OCCURS)...")
    
    with rasterio.open(ORTHO_PATH) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        jobs = [(0, cx, cy, 0.0, "")]
        
        try:
            cleanup_gpu()
            
            def preload_cb(pct, msg):
                print(f"[TEST] Precarga {pct}%: {msg}")
            
            preload_track_gpu(config, jobs, progress_callback=preload_cb)
            print("[TEST] Preload successful!")
            
        except Exception as e:
            print(f"[TEST] !!! PRELOAD FAILED: {e}")
            import traceback
            traceback.print_exc()
            free_pinned_memory()
            return False
    
    # STEP 3: Simulate rendering a frame
    print("[TEST] STEP 3: Rendering a frame...")
    try:
        frame = render_frame_gpu(
            None, [], cx, cy, 45.0, config.width, config.height, 
            config.map_half_width_m, config.arrow_size_px, config.cone_angle_deg,
            config.cone_length_px, config.cone_opacity, config.icon_circle_opacity,
            config.icon_circle_size_px
        )
        frame.get(out=PINNED_BUFFER_GLOBAL)
        print("[TEST] Frame rendered and copied to pinned buffer!")
    except Exception as e:
        print(f"[TEST] Render failed: {e}")
        free_pinned_memory()
        return False
    
    # STEP 4: Cleanup
    print("[TEST] STEP 4: Cleanup...")
    free_pinned_memory()
    cleanup_gpu()
    
    return True

def main():
    print("=" * 60)
    print("[TEST] Simulating FULL app.py render task WITH pinned memory")
    print("=" * 60)
    
    for cycle in range(3):
        print(f"\n{'='*40}")
        print(f"[TEST] CYCLE {cycle+1}/3")
        print(f"{'='*40}")
        
        success = simulate_full_render_task()
        
        if not success:
            print(f"[TEST] FAILED at cycle {cycle+1}")
            return 1
        
        print("[TEST] Waiting 1 second before next cycle...")
        gc.collect()
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("[TEST] ALL CYCLES PASSED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
