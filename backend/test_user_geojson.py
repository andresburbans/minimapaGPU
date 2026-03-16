
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

def test_user_geojson():
    print("[TEST] Starting Test with User GeoJSON and Real Data...")
    
    if not render_gpu.HAS_GPU:
        print("[TEST] ERROR: GPU not available.")
        return

    # User's paths
    GEOJSON_PATH = r"G:\VIDEO-RIVERA\Shapes\Geojsons\FincasBEP.json"
    ORTHO_PATH = Path("gpu_validation") / "test_ortho_crop.tif" # Usamos el que tenemos para el benchmark
    
    if not os.path.exists(GEOJSON_PATH):
        print(f"[TEST] ERROR: {GEOJSON_PATH} not found.")
        return
    if not ORTHO_PATH.exists():
        print(f"[TEST] ERROR: {ORTHO_PATH} not found.")
        return

    W, H = 1920, 1080 # FHD
    
    class Config:
        ortho_path = str(ORTHO_PATH)
        vector_layers = []
        vectors_paths = [GEOJSON_PATH] # El GeoJSON del usuario
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
        icon_circle_opacity = 0.4
        csv_path = "mock.csv"

    config = Config()
    
    with rasterio.open(ORTHO_PATH) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        centers = [(cx, cy), (cx+1, cy+1)] # Pequeño track mock
        
        # Test Loop (Repeat to check for mapping errors)
        for cycle in range(10):
            print(f"\n--- Cycle {cycle+1}/10 ---")
            t0 = time.time()
            
            # Start Preload
            print(f"[TEST] Preloading GPU context with {GEOJSON_PATH}...")
            render_gpu.preload_track_gpu(config, [(0, cx, cy, 0)])
            
            # Render FHD
            print(f"[TEST] Rendering FHD Frame...")
            _ = render_gpu.render_frame_gpu(
                ds, [], cx, cy, 45.0, W, H, 50.0, 40, 60.0, 100, 0.3, 0.5, 20
            )
            
            t1 = time.time()
            print(f"[TEST] Cycle {cycle+1} success. Time: {t1-t0:.2f}s")
            
            # Additional cleanup stress
            print("[TEST] Cleaning up for next cycle...")
            render_gpu._CONTEXT.clear()
            import gc
            gc.collect()
            time.sleep(1) # Extra pause to let driver settle
            
        print("\n[TEST] ALL CYCLES COMPLETED SUCCESSFULLY WITHOUT MAPPING ERRORS.")

if __name__ == "__main__":
    test_user_geojson()
