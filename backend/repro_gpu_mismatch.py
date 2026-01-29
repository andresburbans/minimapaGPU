
import os
import sys
import math
import cupy as cp
import rasterio
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Any

# Ensure backend acts as module root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import render
import render_gpu

# Dummy Config
@dataclass
class VectorLayer:
    name: str = "default"
    path: str = ""
    color: str = "#FF0000"
    width: int = 2
    type: str = "line"
    
    def model_dump(self):
        return self.__dict__

@dataclass
class Config:
    ortho_path: str
    vector_layers: List[VectorLayer]
    vectors_paths: List[str]
    curves_path: str
    line_color: str = "#FF0000"
    line_width: int = 5
    boundary_color: str = "#00FF00"
    boundary_width: int = 5
    point_color: str = "#0000FF"
    map_half_width_m: float = 100.0
    arrow_size_px: int = 50
    cone_length_px: int = 100
    wms_source: str = "google_hybrid"
    
def test_mismatch():
    print("Starting GPU Mismatch Test...")
    
    # Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "gpu_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find a TIF
    tif_files = [f for f in os.listdir(data_dir) if f.endswith(".tif")]
    if not tif_files:
        print("No TIF files found in backend/data. Cannot run test.")
        return
        
    ortho_path = os.path.join(data_dir, tif_files[0])
    print(f"Using Ortho: {ortho_path}")
    
    # Config
    config = Config(
        ortho_path=ortho_path,
        vector_layers=[],
        vectors_paths=[],
        curves_path="",
        map_half_width_m=280.0 # ~560m width
    )
    
    # Define Viewpoint (Center of TIF approx)
    with rasterio.open(ortho_path) as src:
        bounds = src.bounds
        center_e = (bounds.left + bounds.right) / 2
        center_n = (bounds.bottom + bounds.top) / 2
        
        # Shift a bit to test non-centered
        center_e += 500
        center_n += 500
        
        crs = src.crs

    heading = 45.0 # Test rotation
    width = 1920
    height = 1080
    
    # 1. CPU Render
    print("Rendering CPU Frame...")
    with rasterio.open(ortho_path) as src:
        cpu_img = render.render_frame(
            dataset=src,
            vectors=[], # No vectors for basic mismatch test
            center_e=center_e,
            center_n=center_n,
            heading=heading,
            width=width,
            height=height,
            map_half_width_m=config.map_half_width_m,
            arrow_size_px=config.arrow_size_px,
            cone_angle_deg=60,
            cone_length_px=config.cone_length_px,
            cone_opacity=0.5,
            icon_circle_opacity=0.5,
            icon_circle_size_px=20,
            show_compass=True,
            wms_source=config.wms_source
        )
    cpu_path = os.path.join(output_dir, "render_cpu.png")
    cpu_img.save(cpu_path)
    print(f"Saved {cpu_path}")
    
    # 2. GPU Render (Fixed)
    print("Rendering GPU (Fixed) Frame...")
    
    # Preload
    # Fake jobs list with just our point
    jobs = [(0, center_e, center_n)]
    render_gpu.preload_track_gpu(config, jobs)
    
    with rasterio.open(ortho_path) as src:
        gpu_img = render_gpu.render_frame_gpu(
            dataset=src,
            vectors=[],
            center_e=center_e,
            center_n=center_n,
            heading=heading,
            width=width,
            height=height,
            map_half_width_m=config.map_half_width_m,
            arrow_size_px=config.arrow_size_px,
            cone_angle_deg=60,
            cone_length_px=config.cone_length_px,
            cone_opacity=0.5,
            icon_circle_opacity=0.5,
            icon_circle_size_px=20,
            show_compass=True,
            wms_source=config.wms_source
        )
    
    gpu_path = os.path.join(output_dir, "render_gpu_fixed.png")
    gpu_img.save(gpu_path)
    print(f"Saved {gpu_path}")
    
    # 3. Compare
    print("Converting to numpy for Diff...")
    cpu_arr = cp.array(cpu_img)
    gpu_arr = cp.array(gpu_img)
    
    # Convert to float to avoid overflow
    diff = cp.abs(cpu_arr.astype(cp.float32) - gpu_arr.astype(cp.float32))
    mse = cp.mean(diff)
    print(f"Mean Squared Error (Pixel Diff): {mse}")
    
    if mse < 5.0: # arbitrary threshold for "visually similar enough" given different resampling
        print("SUCCESS: GPU Render matches CPU Render!")
    else:
        print("WARNING: Significant difference detected.")
        diff_img = Image.fromarray(cp.asnumpy(diff.astype(cp.uint8)))
        diff_img.save(os.path.join(output_dir, "diff.png"))

if __name__ == "__main__":
    test_mismatch()
