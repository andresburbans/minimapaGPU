
import os
import sys
import time
import subprocess
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import backend.gpu_utils as gpu_utils
print(f"[TEST] GPUs: {gpu_utils.detect_cuda_gpu()}")

try:
    import cupy as cp
    import backend.render_gpu as render_gpu
    from rasterio.transform import Affine
    print(f"[TEST] CuPy Imported. Device: {cp.cuda.Device(0).compute_capability}")
except ImportError as e:
    print(f"[TEST] CuPy Import Failed: {e}")
    sys.exit(1)

def setup_dummy_context():
    """Inject dummy data into render_gpu context to force valid rendering path."""
    print("[TEST] Setting up Dummy Context (16k Texture + Mipmaps)...")
    
    # Create 8k texture (FHD needs high res to simulate load)
    # 8192x8192 RGBA
    h, w = 8192, 8192
    
    # Allocate on GPU
    tex_hwc = cp.random.randint(0, 255, (h, w, 4), dtype=cp.uint8)
    tex_chw = cp.ascontiguousarray(tex_hwc.transpose(2, 0, 1))
    
    render_gpu._CONTEXT.ortho_texture = tex_chw
    render_gpu._CONTEXT.ortho_w = w
    render_gpu._CONTEXT.ortho_h = h
    
    # 0.1m per pixel
    render_gpu._CONTEXT.ortho_res_m = 0.1
    # Identity transform (Pixel 0,0 = 0m, 0m)
    render_gpu._CONTEXT.ortho_transform = Affine(0.1, 0.0, 0.0, 0.0, -0.1, h*0.1)
    render_gpu._CONTEXT.ortho_crs = "EPSG:3857"
    
    # Mipmaps
    print("[TEST] Mipmaps...")
    render_gpu._CONTEXT.mipmaps = [tex_chw]
    # L1
    l1 = cp.ascontiguousarray(tex_chw[:, ::2, ::2])
    render_gpu._CONTEXT.mipmaps.append(l1)
    # L2
    l2 = cp.ascontiguousarray(l1[:, ::2, ::2])
    render_gpu._CONTEXT.mipmaps.append(l2)
    
    render_gpu._CONTEXT.is_ready = True
    print("[TEST] Context Ready.")

def run_test():
    # FHD Output
    W, H = 1920, 1080
    FPS = 30
    DURATION_SEC = 2 # Short run
    TOTAL_FRAMES = FPS * DURATION_SEC
    
    OUTPUT_FILE = r"backend/gpu_validation/pipe_test.mp4"
    if not os.path.exists("backend/gpu_validation"):
        os.makedirs("backend/gpu_validation")
        
    # Setup Context
    try:
        setup_dummy_context()
    except Exception as e:
        print(f"[TEST] Failed to setup context: {e}")
        return

    # Mock Data (Movement across dummy map)
    # Center of map is w/2 * res = 4096 * 0.1 = 409
    cx, cy = 400.0, 400.0

    # Pinned Mem Setup (Zero Copy)
    alloc_size = W * H * 4
    try:
        pinned_mem = cp.cuda.alloc_pinned_memory(alloc_size)
        pinned_buffer = np.frombuffer(pinned_mem, np.uint8)[:alloc_size].reshape((H, W, 4))
    except Exception as e:
        print(f"[TEST] Pinned Memory Failed: {e}")
        return

    # FFmpeg Command
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'rgba',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'h264_nvenc', '-preset', 'p1', '-pix_fmt', 'yuv420p',
        OUTPUT_FILE, '-loglevel', 'error'
    ]

    print(f"[TEST] Starting 3 Cycles (FHD 1080p) WITH HEAVY RENDER...")

    for cycle in range(1, 4):
        print(f"\n--- CYCLE {cycle} ---")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.time()
        
        try:
            for i in range(TOTAL_FRAMES):
                # Render (Valid Context)
                # render_frame_gpu args: dataset, vectors, e, n, heading...
                # We pass None for dataset as context is ready.
                
                # Move center
                ce = cx + i * 0.5
                cn = cy + i * 0.5
                
                gpu_frame = render_gpu.render_frame_gpu(
                    None, [], ce, cn, 45.0 + i, W, H, 100.0, 50, 60.0, 200, 0.3, 0.5, 20,
                    show_compass=True, compass_size_px=40
                )

                # Pinned Transfer
                if hasattr(gpu_frame, 'get'):
                     gpu_frame.get(out=pinned_buffer)
                     proc.stdin.write(pinned_buffer.tobytes())
                else:
                     proc.stdin.write(gpu_frame.tobytes())
                
                if i % 10 == 0:
                    print(f"Cycle {cycle}: {i}/{TOTAL_FRAMES}", end='\r')

            proc.stdin.close()
            proc.wait()
            total_t = time.time() - t0
            print(f"Cycle {cycle}: {TOTAL_FRAMES} frames in {total_t:.2f}s => {TOTAL_FRAMES/total_t:.2f} FPS")
            
        except Exception as e:
            print(f"[TEST] Error: {e}")
            import traceback
            traceback.print_exc()
            proc.kill()

if __name__ == "__main__":
    run_test()
