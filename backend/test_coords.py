
import sys
import os
import math
import numpy as np
from rasterio.transform import Affine

sys.path.append(os.path.join(os.path.dirname(__file__)))
import render_gpu
try:
    import cupy as cp
except:
    print("No CuPy")
    sys.exit(0)

def test_transform():
    print("Testing Transform Logic...")
    
    # Mock Texture: 1000x1000
    h, w = 1000, 1000
    tex_chw = cp.zeros((4, h, w), dtype=cp.uint8)
    # Fill with RED
    tex_chw[0, :, :] = 255
    tex_chw[3, :, :] = 255
    
    # Transform: 1 px = 1 meter. Top-Left at (0, 1000).
    # (0,0) px -> (0, 1000) m
    # (0, 1000) px -> (0, 0) m
    tf = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1000.0)
    
    # We want to sample at Center (500m, 500m).
    # Should correspond to pixel (500, 500).
    ce, cn = 500.0, 500.0
    heading = 0.0
    m_per_px = 1.0
    out_w, out_h = 100, 100
    
    # Expected result: Red Square.
    res = render_gpu._sample_using_inverse_transform(
        texture_planar=tex_chw,
        center_e=ce,
        center_n=cn,
        heading=heading,
        m_per_px_out=m_per_px,
        out_h=out_h,
        out_w=out_w,
        ortho_transform=tf,
        mipmap_level=0
    )
    
    # Res is (H, W, 4)
    # Check center pixel
    center_px = res[out_h//2, out_w//2]
    print(f"Sampled Center Pixel: {center_px} (Expected [255, 0, 0, 255])")
    
    if center_px[0] == 255 and center_px[3] == 255:
        print("PASS: Coordinates map correctly.")
    else:
        print("FAIL: Coordinates incorrect.")
        
        # Debug Output
        # Calculate expected Offset
        itf = ~tf
        cx, cy = itf * (ce, cn)
        print(f"Inverse Transform Center: ({cx}, {cy}) (Expected 500, 500)")
        
if __name__ == "__main__":
    test_transform()
