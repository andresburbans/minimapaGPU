
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import numpy as np
import os
import sys
from pathlib import Path

# Add backend to path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

import gpu_utils
gpu_utils.fix_cuda_dll_paths()

def test_fused_correctness():
    print("Testing Fused Affine Transform Correctness...")
    
    # Create random 4-channel image
    H, W = 512, 512
    data = cp.random.randint(0, 256, (4, H, W), dtype=cp.uint8)
    
    # 2D Affine parameters
    matrix_2d = cp.array([[0.8, 0.2], [-0.1, 0.9]], dtype=cp.float32)
    offset_2d = cp.array([10.5, 20.3], dtype=cp.float32)
    
    # Method 1: Loop over channels (Old)
    expected = cp.zeros((4, H, W), dtype=cp.uint8)
    for i in range(4):
        ndimage.affine_transform(data[i], matrix_2d, offset=offset_2d, output=expected[i], order=1)
    
    # Method 2: Fused 3x3 pass (New)
    print("Preparing 3x3 matrix...")
    matrix_3x3 = cp.eye(3, dtype=cp.float32)
    matrix_3x3[1:3, 1:3] = matrix_2d
    offset_3 = cp.array([0.0, float(offset_2d[0]), float(offset_2d[1])], dtype=cp.float32)
    
    print("Running fused transform...")
    actual = cp.zeros((4, H, W), dtype=cp.uint8)
    ndimage.affine_transform(data, matrix_3x3, offset=offset_3, output=actual, order=1)
    
    print("Comparison...")
    diff = cp.abs(expected.astype(cp.int16) - actual.astype(cp.int16))
    max_diff = float(cp.max(diff).get())
    mean_diff = float(cp.mean(diff).get())
    
    print(f"Max Difference: {max_diff}")
    print(f"Mean Difference: {mean_diff}")
    
    if max_diff <= 1: # Allow for tiny rounding diffs if any
        print("✅ SUCCESS: Fused transform matches loop-over-channels.")
    else:
        print(f"❌ FAILURE: Results mismatch! Max diff: {max_diff}")
        sys.exit(1)

if __name__ == "__main__":
    test_fused_correctness()
