
import os
import sys
import subprocess
import time
import rasterio
from PIL import Image
from dataclasses import dataclass
from typing import List

# Mock Config
@dataclass
class VectorLayer:
    model_dump = lambda self: {}

@dataclass
class Config:
    ortho_path: str
    width: int
    height: int
    fps: int
    vector_layers: list
    vectors_paths: list
    curves_path: str
    line_color: str = "#FF0000"
    line_width: int = 5
    boundary_color: str = "#00FF00"
    boundary_width: int = 5
    point_color: str = "#0000FF"
    map_half_width_m: float = 100.0
    arrow_size_px: int = 50
    cone_length_px: int = 100
    cone_angle_deg: float = 60
    cone_opacity: float = 0.5
    icon_circle_opacity: float = 0.5
    icon_circle_size_px: int = 20
    show_compass: bool = True
    compass_size_px: int = 40
    wms_source: str = "google_hybrid"
    use_gpu: bool = True

def test_pipe():
    print("Starting GPU Pipe Test...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "gpu_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    ortho_path = os.path.join(output_dir, "test_ortho_crop.tif")
    if not os.path.exists(ortho_path):
        print(f"Ortho not found: {ortho_path}")
        return

    # Check Render GPU avail
    try:
        import render_gpu
        if not render_gpu.init_gpu()["available"]:
            print("GPU not available.")
            return
    except ImportError:
        print("render_gpu not found")
        return

    config = Config(
        ortho_path=ortho_path,
        width=1280,
        height=720,
        fps=30,
        vector_layers=[],
        vectors_paths=[],
        curves_path=""
    )
    
    # 3 sec video = 90 frames
    total_frames = 90
    
    output_video = os.path.join(output_dir, "pipe_test.mp4")
    
    # FFmpeg Cmd
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{config.width}x{config.height}",
        "-pix_fmt", "rgba",
        "-r", str(config.fps),
        "-i", "-",
        "-c:v", "h264_nvenc", # Force NVENC for test
        "-preset", "p1", # Fastest
        "-pix_fmt", "yuv420p",
        output_video
    ]
    
    print(f"Running FFmpeg: {' '.join(cmd)}")
    
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10**7)
    except Exception as e:
        print(f"Failed to start ffmpeg: {e}")
        return

    # Preload
    jobs = [(i, 0, 0) for i in range(total_frames)] # Dummy
    # Need real centers
    with rasterio.open(ortho_path) as src:
        cx, cy = (src.bounds.left + src.bounds.right)/2, (src.bounds.bottom + src.bounds.top)/2
        # Move center 1000m East (1km)
        centers = []
        for i in range(total_frames):
            centers.append((i, cx + i*10, cy))
        
        # Preload
        render_gpu.preload_track_gpu(config, [(0, cx, cy), (0, cx+1000, cy)])
        
        start_t = time.time()
        for i in range(total_frames):
            idx, ce, cn = centers[i]
            img = render_gpu.render_frame_gpu(
                dataset=src,
                vectors=[],
                center_e=ce,
                center_n=cn,
                heading=i * 2, # Rotate
                width=config.width,
                height=config.height,
                map_half_width_m=config.map_half_width_m,
                arrow_size_px=config.arrow_size_px,
                cone_angle_deg=config.cone_angle_deg,
                cone_length_px=config.cone_length_px,
                cone_opacity=config.cone_opacity,
                icon_circle_opacity=config.icon_circle_opacity,
                icon_circle_size_px=config.icon_circle_size_px,
                show_compass=config.show_compass,
                wms_source=config.wms_source
            )
            
            # Pipe
            try:
                proc.stdin.write(img.tobytes())
            except Exception as e:
                print(f"Pipe write error: {e}")
                break
                
        proc.stdin.close()
        proc.wait()
        
        end_t = time.time()
        dur = end_t - start_t
        fps = total_frames / dur
        print(f"Finished {total_frames} frames in {dur:.2f}s ({fps:.2f} FPS)")
        
        if os.path.exists(output_video) and os.path.getsize(output_video) > 1000:
            print("SUCCESS: Video created.")
        else:
            print("FAILURE: Video missing or empty.")

if __name__ == "__main__":
    test_pipe()
