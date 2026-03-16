import cupy as cp
import cupyx.scipy.ndimage as ndimage
import time
import os
import sys

# Simplified environment fix
if os.name == 'nt':
    dll_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin",
        os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "cupy", ".data", "lib"),
        os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "cupy_backends", "cuda", "libs"),
    ]
    for p in dll_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

def test_strided_output():
    H, W = 1080, 1920
    texture_planar = cp.random.randint(0, 255, (4, H, W), dtype=cp.uint8)
    
    matrix = cp.array([[1.0, 0.0], [0.0, 1.0]], dtype=cp.float32)
    offset = cp.array([0.0, 0.0], dtype=cp.float32)
    
    # 1. PLANAR ALLOCATION (Current Method)
    start = time.time()
    result_planar = cp.zeros((4, H, W), dtype=cp.uint8)
    for i in range(4):
        ndimage.affine_transform(
            texture_planar[i], matrix, offset=offset, output_shape=(H, W),
            output=result_planar[i], order=1, mode='constant', cval=0, prefilter=False
        )
    final_1 = cp.ascontiguousarray(cp.transpose(result_planar, (1, 2, 0)))
    cp.cuda.Stream.null.synchronize()
    msg_1 = f"Planar+Transpose:_ {time.time() - start:.6f}s"
    print(msg_1)
    
    # 2. INTERLEAVED PRE-ALLOCATION (Strided Output)
    start = time.time()
    final_2 = cp.zeros((H, W, 4), dtype=cp.uint8) # C-contiguous (H, W, 4)
    # final_2[:, :, 0] is strided!
    for i in range(4):
        ndimage.affine_transform(
            texture_planar[i], matrix, offset=offset, output_shape=(H, W),
            output=final_2[:, :, i], # STRIDED WRITE
            order=1, mode='constant', cval=0, prefilter=False
        )
    cp.cuda.Stream.null.synchronize()
    msg_2 = f"Direct Strided:__ {time.time() - start:.6f}s"
    print(msg_2)
    
    # Check correctness
    diff = cp.abs(final_1.astype(cp.float32) - final_2.astype(cp.float32)).sum()
    print(f"Difference: {diff}")
    
    if diff == 0:
        print("OPTIMIZATION VALID: Strided write works and is correct.")
    else:
        print("WARNING: Output mismatch.")

if __name__ == "__main__":
    try:
        test_strided_output()
    except Exception as e:
        print(f"Error: {e}")
