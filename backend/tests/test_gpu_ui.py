
import cupy as cp
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add backend to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

import gpu_utils
gpu_utils.fix_cuda_dll_paths()
from render_gpu import _draw_ui_gpu, _alpha_composite_gpu

def test_gpu_ui():
    print("Testing GPU UI Composition...")
    
    W, H = 1000, 600
    base = cp.zeros((H, W, 4), dtype=cp.uint8)
    base[:, :, 2] = 50 # Dark blue background
    base[:, :, 3] = 255
    
    # Create fake cone (red square for visibility)
    cone = cp.zeros((H, W, 4), dtype=cp.uint8)
    cone[100:300, 400:600] = cp.array([255, 0, 0, 100], dtype=cp.uint8)
    
    # Create fake icon (green circle)
    icon = cp.zeros((100, 100, 4), dtype=cp.uint8)
    # Draw simple square in CP
    icon[20:80, 20:80] = cp.array([0, 255, 0, 255], dtype=cp.uint8)
    
    # Run UI drawer
    result = _draw_ui_gpu(base.copy(), cone, icon, W, H)
    
    # Move to CPU for verification
    res_cpu = cp.asnumpy(result)
    
    # Check center location (icon)
    center_pixel = res_cpu[H//2, W//2]
    print(f"Center pixel color (expect green): {center_pixel}")
    
    # Check cone area
    cone_pixel = res_cpu[200, 500]
    print(f"Cone area pixel color (expect partly red): {cone_pixel}")
    
    # Basic assertions
    assert center_pixel[1] == 255, "Icon not rendered correctly at center"
    assert cone_pixel[0] > 0, "Cone mask not alpha-composited"
    
    print("✅ SUCCESS: GPU UI composition logic verified.")
    
    # Save result for visual check if needed
    Image.fromarray(res_cpu).save(str(ROOT / "test_gpu_ui_output.png"))

if __name__ == "__main__":
    test_gpu_ui()
