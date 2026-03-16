"""
Test that simulates the EXACT scenario: Preview → then → Render
This is what happens when user generates preview then clicks render.
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

def setup_config():
    ORTHO_PATH = Path("gpu_validation") / "test_ortho_crop.tif"
    GEOJSON_PATH = r"G:\VIDEO-RIVERA\Shapes\Geojsons\FincasBEP.json"
    
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
    
    return MockConfig(), ORTHO_PATH

def simulate_preview():
    """Simulates the /preview endpoint"""
    print("\n[TEST] === SIMULATING PREVIEW ===")
    config, ortho_path = setup_config()
    
    if not ortho_path.exists():
        print(f"[TEST] ERROR: {ortho_path} not found.")
        return False
    
    with rasterio.open(ortho_path) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        jobs = [(0, cx, cy, 0.0, "")]
        
        # This is what /preview does
        print("[TEST] Preview: cleanup_gpu()...")
        cleanup_gpu()
        
        print("[TEST] Preview: preload_track_gpu()...")
        try:
            preload_track_gpu(config, jobs)
        except Exception as e:
            print(f"[TEST] Preview preload failed: {e}")
            return False
        
        print("[TEST] Preview: render_frame_gpu()...")
        try:
            frame = render_frame_gpu(
                ds, [], cx, cy, 45.0, config.width, config.height, 
                config.map_half_width_m, config.arrow_size_px, config.cone_angle_deg,
                config.cone_length_px, config.cone_opacity, config.icon_circle_opacity,
                config.icon_circle_size_px
            )
            # Convert to PIL like preview does
            frame_np = frame.get()
            print(f"[TEST] Preview rendered: {frame_np.shape}")
        except Exception as e:
            print(f"[TEST] Preview render failed: {e}")
            return False
    
    print("[TEST] Preview complete!")
    return True

def simulate_render():
    """Simulates the /render endpoint (the _render_task function)"""
    print("\n[TEST] === SIMULATING RENDER TASK ===")
    config, ortho_path = setup_config()
    
    if not ortho_path.exists():
        print(f"[TEST] ERROR: {ortho_path} not found.")
        return False
    
    with rasterio.open(ortho_path) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        jobs = [(0, cx, cy, 0.0, "")]
        
        # This is what _render_task does
        print("[TEST] Render: cleanup_gpu()...")
        cleanup_gpu()
        
        print("[TEST] Render: preload_track_gpu() - THIS IS WHERE ERROR OCCURS...")
        try:
            def cb(pct, msg):
                print(f"[TEST] Precarga {pct}%: {msg}")
            preload_track_gpu(config, jobs, progress_callback=cb)
        except Exception as e:
            print(f"[TEST] !!! RENDER PRELOAD FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Allocate pinned memory like app.py does
        print("[TEST] Render: Allocating pinned memory...")
        alloc_size = config.width * config.height * 4
        try:
            pinned_mem = cp.cuda.alloc_pinned_memory(alloc_size)
            pinned_buffer = np.frombuffer(pinned_mem, np.uint8)[:alloc_size].reshape((config.height, config.width, 4))
        except Exception as e:
            print(f"[TEST] Pinned memory failed: {e}")
            return False
        
        print("[TEST] Render: render_frame_gpu()...")
        try:
            frame = render_frame_gpu(
                ds, [], cx, cy, 45.0, config.width, config.height, 
                config.map_half_width_m, config.arrow_size_px, config.cone_angle_deg,
                config.cone_length_px, config.cone_opacity, config.icon_circle_opacity,
                config.icon_circle_size_px
            )
            frame.get(out=pinned_buffer)
            print(f"[TEST] Render frame complete: {pinned_buffer.shape}")
        except Exception as e:
            print(f"[TEST] Render frame failed: {e}")
            return False
        
        # Cleanup
        pinned_buffer = None
        pinned_mem = None
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    print("[TEST] Render complete!")
    return True

def main():
    print("=" * 60)
    print("[TEST] Simulating: Preview → Render (exact user flow)")
    print("=" * 60)
    
    # Cycle 1: Preview then Render
    print("\n" + "=" * 40)
    print("[TEST] CYCLE 1: Preview → Render")
    print("=" * 40)
    
    if not simulate_preview():
        print("[TEST] FAILED at Preview")
        return 1
    
    time.sleep(1)  # Small pause like user would have
    
    if not simulate_render():
        print("[TEST] FAILED at Render after Preview")
        return 1
    
    # Cycle 2: Another round
    print("\n" + "=" * 40)
    print("[TEST] CYCLE 2: Preview → Render (again)")
    print("=" * 40)
    
    if not simulate_preview():
        print("[TEST] FAILED at Preview (cycle 2)")
        return 1
    
    time.sleep(1)
    
    if not simulate_render():
        print("[TEST] FAILED at Render after Preview (cycle 2)")
        return 1
    
    print("\n" + "=" * 60)
    print("[TEST] ALL CYCLES PASSED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
