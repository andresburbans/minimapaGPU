
import os
import sys
import time
import numpy as np
import rasterio
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import backend.gpu_utils as gpu_utils
import backend.render_gpu as render_gpu

def run_real_perf_test():
    print("[BENCHMARK] Starting Real Data Performance Test...")
    
    if not render_gpu.HAS_GPU:
        print("[BENCHMARK] ERROR: GPU not available.")
        return

    # Paths
    VAL_DIR = Path("backend/gpu_validation")
    ORTHO_PATH = VAL_DIR / "test_ortho_crop.tif"
    VECTORS = [
        (VAL_DIR / "LinderoGeneral.geojson", "red", 2),
        (VAL_DIR / "Vias.geojson", "blue", 2)
    ]
    
    if not ORTHO_PATH.exists():
        print(f"[BENCHMARK] ERROR: {ORTHO_PATH} not found.")
        return

    # FHD
    W, H = 1920, 1080
    NUM_FRAMES = 30
    
    # 1. Preload
    print("[BENCHMARK] Preloading real data...")
    t_start_preload = time.time()
    
    # Mock some center points (circular path)
    with rasterio.open(ORTHO_PATH) as ds:
        bounds = ds.bounds
        cx, cy = (bounds.left + bounds.right)/2, (bounds.bottom + bounds.top)/2
        radius = (bounds.right - bounds.left) * 0.1
        centers = []
        for i in range(NUM_FRAMES):
            angle = 2 * np.pi * i / NUM_FRAMES
            centers.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
            
        # Mock config
        class Config:
            ortho_path = str(ORTHO_PATH)
            vector_layers = []
            vectors_paths = [str(VECTORS[0][0]), str(VECTORS[1][0])]
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
            
        config = Config()
        
        # We need to load vectors properly as render_gpu.preload_track_gpu does
        import backend.render as render
        vec_data = render.load_vectors(
            ds.crs, [], config.vectors_paths, None, "red", 2, "blue", 2, "green"
        )
        
        render_gpu._CONTEXT.clear()
        render_gpu._CONTEXT.preload(
            ds, centers, config.map_half_width_m * 2.5, vec_data
        )
        
    t_preload = time.time() - t_start_preload
    print(f"[BENCHMARK] Preload finished in {t_preload:.2f}s")
    
    # 2. Render Loop
    print(f"[BENCHMARK] Rendering {NUM_FRAMES} frames @ FHD...")
    t_start_render = time.time()
    
    for i in range(NUM_FRAMES):
        ce, cn = centers[i]
        heading = (i * 10) % 360
        _ = render_gpu.render_frame_gpu(
            None, [], ce, cn, heading, W, H, 50.0, 40, 60.0, 100, 0.3, 0.5, 20,
            show_compass=True, compass_size_px=40
        )
        if i % 5 == 0:
            print(f"Frame {i}/{NUM_FRAMES}", end='\r')
            
    t_render = time.time() - t_start_render
    fps = NUM_FRAMES / t_render
    
    print(f"\n[BENCHMARK] Results:")
    print(f"Total Time: {t_render:.2f}s")
    print(f"FPS: {fps:.2f}")
    
if __name__ == "__main__":
    run_real_perf_test()
