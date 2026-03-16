"""
Test that simulates the EXACT flow that app.py uses when it triggers preload.
This should help reproduce the cudaErrorAlreadyMapped error.
"""
import os
import sys
import time
import gc

# Simulate being in the backend directory like uvicorn would
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import exactly like app.py does
import gpu_utils  # This sets CUDA_VISIBLE_DEVICES

# Simulate the conditional import from app.py
GPU_RENDER_AVAILABLE = False
try:
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

def simulate_app_preload():
    """Simulates exactly what app.py does during render task"""
    
    ORTHO_PATH = Path("gpu_validation") / "test_ortho_crop.tif"
    GEOJSON_PATH = r"G:\VIDEO-RIVERA\Shapes\Geojsons\FincasBEP.json"
    
    if not ORTHO_PATH.exists():
        print(f"[TEST] ERROR: {ORTHO_PATH} not found.")
        return False
    
    # Mock config exactly like RenderConfig in app.py
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
    
    with rasterio.open(ORTHO_PATH) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        
        # Create mock jobs like app.py does
        jobs = [(0, cx, cy, 0.0, "")]
        
        # This is EXACTLY what app.py _render_task does:
        print("[TEST] Simulating _render_task GPU preload...")
        
        if config.use_gpu and GPU_RENDER_AVAILABLE:
            try:
                print("[TEST] Calling cleanup_gpu()...")
                cleanup_gpu()
                
                def preload_cb(pct, msg):
                    print(f"[TEST] Precarga {pct}%: {msg}")
                
                print("[TEST] Calling preload_track_gpu()...")
                preload_track_gpu(config, jobs, progress_callback=preload_cb)
                print("[TEST] Preload successful!")
                return True
                
            except Exception as e:
                print(f"[TEST] !!! PRELOAD FAILED: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    return True

def main():
    print("=" * 60)
    print("[TEST] Simulating app.py GPU preload flow")
    print("=" * 60)
    
    # Run multiple cycles like the web app would
    for cycle in range(5):
        print(f"\n{'='*40}")
        print(f"[TEST] CYCLE {cycle+1}/5")
        print(f"{'='*40}")
        
        success = simulate_app_preload()
        
        if not success:
            print(f"[TEST] FAILED at cycle {cycle+1}")
            return 1
        
        # Simulate time between requests
        print("[TEST] Waiting 2 seconds before next cycle...")
        gc.collect()
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("[TEST] ALL CYCLES PASSED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
