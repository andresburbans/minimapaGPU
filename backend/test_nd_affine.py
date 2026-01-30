import cupy as cp
import cupyx.scipy.ndimage as ndimage
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
import gpu_utils
gpu_utils.fix_cuda_dll_paths()

def test_3d_affine():
    # Create a 3D array (Channels, Height, Width)
    data = cp.zeros((4, 100, 100), dtype=cp.uint8)
    data[0, 10:20, 10:20] = 255 # Channel 0 has a square
    data[1, 30:40, 30:40] = 128 # Channel 1 has a square
    
    # 2x2 Matrix (Rotation 45 deg)
    matrix = cp.array([[0.707, -0.707], [0.707, 0.707]], dtype=cp.float32)
    offset = cp.array([50, 50], dtype=cp.float32) # Offset for rows/cols
    
    print("Attempting 3D affine with 2D matrix...")
    try:
        # We want to transform only H, W. 
        # offset should probably be 3D if input is 3D? Or just for the transformed axes?
        # In current CuPy, if matrix is (2,2), it might error if input is 3D.
        
        # Test Case A: Matrix (2,2), Offset (2,)
        res = ndimage.affine_transform(data, matrix, offset=offset, output_shape=(100, 100))
        print("Success A: Result shape", res.shape)
    except Exception as e:
        print("Failed A:", e)

    try:
        # Test Case B: Using a 3x3 matrix where first dim is identity for channels
        matrix3 = cp.eye(3)
        matrix3[1:3, 1:3] = matrix
        offset3 = cp.array([0, 50, 50])
        res = ndimage.affine_transform(data, matrix3, offset=offset3)
        print("Success B: Result shape", res.shape)
    except Exception as e:
        print("Failed B:", e)

if __name__ == "__main__":
    test_3d_affine()
