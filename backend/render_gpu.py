
import math
import logging
import io
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform
from rasterio.transform import Affine
from PIL import Image, ImageDraw, ImageOps

import gpu_utils 

# Try importing GPU libraries
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("[GPU] CuPy not found, falling back to CPU logic.")

try:
    from render import (
        Segment,
        _to_rgba,
        _normalize_rgba,
        _draw_geometry_precise,
        _load_nav_icon,
        _draw_compass,
        _draw_cone,
        _draw_center_icon,
        load_vectors,
        clip_vectors,
        _GLOBAL_NAV_ICON,
        _fetch_wms_mosaic_for_bounds,
        _latlon_to_pixel,
        _rotate_image,
        _clamp_latlon,
    )
except ImportError:
    pass

# Constants
_ROTATION_OVERSCAN = 1.5

def init_gpu():
    """Check and return GPU status."""
    if not HAS_GPU:
        return {"available": False, "backend": "cpu"}
    try:
        cnt = cp.cuda.runtime.getDeviceCount()
        if cnt == 0:
            return {"available": False, "error": "No CUDA devices found"}
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get('name', b'Unknown').decode('utf-8')
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return {
            "available": True, 
            "count": cnt, 
            "backend": "cupy", 
            "device": name,
            "memory_free": free_mem,
            "memory_total": total_mem
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

class GPURenderContext:
    """
    STRONG GPU CONTEXT:
    Manages Ortho, WMS, and Vector layers as cached GPU textures.
    Ensures zero-jitter by sharing coordinate systems.
    """
    def __init__(self):
        self.ortho_texture = None # (RGBA) GPU array
        self.ortho_transform = None
        self.ortho_crs = None
        self.ortho_nodata = None
        
        # New: Baked Vector Layer on GPU
        self.vector_texture = None # (RGBA) GPU array
        self.has_vectors = False
        
        self.wms_texture = None # (RGBA) GPU array
        self.wms_bounds_px = None 
        self.wms_zoom = None
        
        # New: Cached Icon Texture
        self.icon_texture = None # (RGBA, small)
        self.cone_texture = None # (RGBA, small)
        
        self.last_config_key = None
        self.is_ready = False

    def clear(self):
        """Free GPU memory and reset context."""
        self.ortho_texture = None
        self.vector_texture = None
        self.wms_texture = None
        self.icon_texture = None
        self.cone_texture = None
        self.is_ready = False
        if HAS_GPU:
            try:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                print("[GPU] Context cleared and memory freed.")
            except Exception as e:
                print(f"[GPU] Error clearing memory: {e}")

    def preload(self, dataset, center_points, margin_m, vectors=None, arrow_size=100, cone_len=200, wms_source="google_hybrid"):
        """
        Loads Ortho, Vectors, and WMS into GPU memory.
        """
        if not HAS_GPU: return False
        
        print("[GPU] Starting Heavy Preload (99% Mode)...")
        
        # --- 1. Bounds Calculation ---
        es = [p[0] for p in center_points]
        ns = [p[1] for p in center_points]
        xmin, xmax = min(es) - margin_m, max(es) + margin_m
        ymin, ymax = min(ns) - margin_m, max(ns) + margin_m
        
        # --- 2. Load Ortho (High Res) ---
        window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
        
        # Optimize size for GPU memory (aim for ~4GB usage max or 95% of available)
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        # Reserve 1GB for overhead, use rest for textures
        avail_mem = max(free_mem - 1024*1024*1024, 1024*1024*512)
        
        # Calculate max dimension
        # 4 channels * 1 byte (uint8) = 4 bytes/pixel 
        # (We load as uint8 directly to save space, assuming normalize on CPU or block-wise)
        max_pixels = avail_mem / 8 # Factor of safety for mips/vars
        side = int(math.sqrt(max_pixels))
        side = min(side, 16384) # Cap at 16k texture
        
        w_m = xmax - xmin
        h_m = ymax - ymin
        ratio = w_m / h_m
        if ratio > 1:
            tw, th = side, int(side / ratio)
        else:
            tw, th = int(side * ratio), side
            
        print(f"[GPU] Allocating Textures: {tw}x{th} (~{tw*th*4/1024/1024:.0f} MB each)")

        # Read Ortho
        # Read as byte directly if possible to save RAM? rasterio reads as type. 
        # reading boundless with fill_value
        cpu_data = dataset.read(
            window=window,
            out_shape=(dataset.count, th, tw),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=dataset.nodata
        )
        
        # Store transform
        base_transform = rasterio.windows.transform(window, dataset.transform)
        scale_x = window.width / tw
        scale_y = window.height / th
        self.ortho_transform = base_transform * Affine.scale(scale_x, scale_y)
        self.ortho_crs = dataset.crs
        
        # Upload Ortho
        self.ortho_texture = _normalize_gpu_rgba(cp.asarray(cpu_data), dataset.nodata)
        del cpu_data # Free RAM
        
        # --- 3. Bake Vectors (Zero Jitter) ---
        if vectors:
            print("[GPU] Baking Vectors to Texture...")
            # We use PIL to draw vectors onto a transparent image of SAME SIZE as Ortho
            # This guarantees perfect alignment.
            vec_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            draw = ImageDraw.Draw(vec_img)
            
            # Helper: Geo -> Pixel (in this specific texture)
            # Texture Coord T = ~Transform * Geo
            inv_tf = ~self.ortho_transform
            
            def gpu_vec_transform(e, n):
                # Apply inverse transform
                # x = (e - c) / a ... simplified by Affine logic
                c, r = inv_tf * (e, n)
                return c, r

            for geom_iter, color, width, pattern in vectors:
                # Scale width slightly because we might zoom in/out
                # But here we bake at fixed resolution. 
                # If we zoom in 2x, this line will look 2x thicker. 
                # This is acceptable for stability.
                # Or we can draw slightly thicker lines.
                eff_width = max(1, int(width * (tw / window.width) * 2)) # Heuristic
                
                for geom in geom_iter:
                    _draw_geometry_precise(draw, geom, gpu_vec_transform, color, eff_width, pattern)
            
            # Upload Vectors
            self.vector_texture = cp.asarray(np.array(vec_img))
            self.has_vectors = True
            
        # --- 4. WMS Background ---
        # Same Logic as before but simplified
        w_geo, s_geo, e_geo, n_geo = transform_bounds(dataset.crs, "EPSG:4326", xmin, ymin, xmax, ymax)
        
        # Smart Zoom Selection
        span_deg = max(e_geo - w_geo, n_geo - s_geo)
        # 16k pixels for span_deg
        # 360 deg = 256 * 2^z pixels
        # 2^z = (pixels * 360) / (256 * span)
        if span_deg > 0:
            ideal_z = math.log2((max(tw,th) * 360) / (256 * span_deg))
            wms_zoom = min(19, max(10, int(ideal_z)))
        else:
            wms_zoom = 18
            
        print(f"[GPU] Fetching WMS Level {wms_zoom} (Source: {wms_source})...")
        ret_wms = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, wms_zoom, source=wms_source)
        wms_img = ret_wms[0]
        wms_info = ret_wms[1]
        
        if len(wms_info) >= 2:
            self.wms_bounds_px = (wms_info[0], wms_info[1])
            self.wms_zoom = wms_zoom
            # Ensure 4 channels
            self.wms_texture = cp.asarray(np.array(wms_img.convert("RGBA")))
        
        # --- 5. Bake Icons ---
        # Generate Icon Texture
        # We assume standard arrow
        icon_sz = int(arrow_size * 1.5)
        ic_img = Image.new("RGBA", (icon_sz, icon_sz), (0,0,0,0))
        _draw_center_icon(ic_img, (icon_sz//2, icon_sz//2), arrow_size, 0.4, arrow_size//3, 0)
        self.icon_texture = cp.asarray(np.array(ic_img))
        
        self.is_ready = True
        return True

_CONTEXT = GPURenderContext()

def _normalize_gpu_rgba(arr_gpu, nodata=None):
    """Normalize and convert to (H, W, 4) uint8"""
    c, h, w = arr_gpu.shape
    if c >= 3:
        rgb = arr_gpu[:3]
    else:
        rgb = cp.repeat(arr_gpu[0][None, ...], 3, axis=0) # Grey->RGB

    # Alpha
    if c == 4:
        alpha = arr_gpu[3]
    elif nodata is not None:
        if cp.isnan(float(nodata)):
            mask = ~cp.isnan(arr_gpu[0])
        else:
            mask = arr_gpu[0] != nodata
        alpha = (mask.astype(cp.uint8) * 255)
    else:
        alpha = cp.full((h, w), 255, dtype=cp.uint8)

    # Normalize RGB (Simple MinMax)
    rgb_f = rgb.astype(cp.float32)
    # Fast approach: use global min/max or static
    # Using 2%-98% on GPU
    v_min = 0 # cp.percentile(rgb_f, 2) # Percentile is slow?
    v_max = 255 # cp.percentile(rgb_f, 98)
    
    # Let's trust standard byte range for speed unless it looks weird
    # If 16-bit or float, we MUST normalize
    if arr_gpu.dtype != cp.uint8:
         v_min = float(cp.min(rgb_f))
         v_max = float(cp.max(rgb_f))
         if v_max > v_min:
             rgb_u8 = ((rgb_f - v_min) / (v_max - v_min) * 255).astype(cp.uint8)
         else:
             rgb_u8 = cp.zeros_like(rgb, dtype=cp.uint8)
    else:
         rgb_u8 = rgb

    return cp.dstack((rgb_u8[0], rgb_u8[1], rgb_u8[2], alpha))

def _affine_sample(texture, matrix, offset, out_h, out_w):
    """
    Core GPU sampler.
    texture: (H, W, 4)
    matrix: 2x2 [[m00, m01], [m10, m11]]
    offset: [oy, ox]
    """
    # Prepare output
    output = cp.zeros((out_h, out_w, 4), dtype=cp.uint8)
    
    # Transpose for affine function: (H, W, C) -> (C, H, W)
    tex_c = cp.transpose(texture, (2, 0, 1))
    
    # Split channels to avoid 4D affine (cupyx limitation)
    for i in range(4):
        ndimage.affine_transform(
            tex_c[i],
            matrix,
            offset=offset,
            output_shape=(out_h, out_w),
            output=output[:, :, i], # Write direct to channel
            order=1, # Bilinear
            mode='constant',
            cval=0,
            prefilter=False # Faster
        )
    return output

def _blit_icon(bg_arr, icon_arr, x, y, angle_deg):
    """
    Blits rotated icon onto bg_arr (H,W,4).
    This is a naive GPU implementation.
    """
    # For speed, we just rotate the icon on CPU via PIL if it's small? 
    # No, we promised 99% GPU.
    # But rotating a 100x100 patch on GPU is easy.
    
    ih, iw, _ = icon_arr.shape
    bh, bw, _ = bg_arr.shape
    
    # Rotate Icon
    # Angle needs to be negative for image coord system
    # We can use ndimage.rotate
    icon_rot = ndimage.rotate(icon_arr, angle_deg, axes=(0,1), reshape=True, order=1, prefilter=False)
    rh, rw, _ = icon_rot.shape
    
    # Coords
    lx = int(x - rw//2)
    ty = int(y - rh//2)
    
    # Clip
    sx, sy = 0, 0
    ex, ey = rw, rh
    
    if lx < 0: sx = -lx; lx = 0
    if ty < 0: sy = -ty; ty = 0
    if lx + (ex-sx) > bw: ex -= (lx + (ex-sx)) - bw
    if ty + (ey-sy) > bh: ey -= (ty + (ey-sy)) - bh
    
    if ex <= sx or ey <= sy: return bg_arr
    
    # Alpha Blend Patch
    patch_bg = bg_arr[ty:ty+(ey-sy), lx:lx+(ex-sx)].astype(cp.float32)
    patch_ic = icon_rot[sy:ey, sx:ex].astype(cp.float32)
    
    alpha = patch_ic[:,:,3:4] / 255.0
    
    out_patch = patch_ic[:,:,:3] * alpha + patch_bg[:,:,:3] * (1.0 - alpha)
    out_alpha = cp.maximum(patch_bg[:,:,3], patch_ic[:,:,3]) # Simple max alpha
    
    bg_arr[ty:ty+(ey-sy), lx:lx+(ex-sx), :3] = out_patch.astype(cp.uint8)
    bg_arr[ty:ty+(ey-sy), lx:lx+(ex-sx), 3] = out_alpha.astype(cp.uint8)
    
    return bg_arr

def render_frame_gpu(
    dataset: rasterio.io.DatasetReader,
    vectors: List[Tuple[Iterable, str, int, str]],
    center_e: float,
    center_n: float,
    heading: float,
    width: int,
    height: int,
    map_half_width_m: float,
    arrow_size_px: int,
    cone_angle_deg: float,
    cone_length_px: int,
    cone_opacity: float,
    icon_circle_opacity: float,
    icon_circle_size_px: int,
    compass_size_px: int = 40,
    use_context: bool = True,
    wms_source: str = "google_hybrid",
) -> Image.Image:
    
    if not HAS_GPU:
        raise RuntimeError("GPU acceleration requested but CuPy is not available.")

    # Matrix Setup
    # Target: Output grid (width, height) centered at center_e, center_n
    # Meters per pixel
    m_per_px_out = (map_half_width_m * 2.0) / width
    
    # Rotation Angle (-heading)
    theta = math.radians(-heading)
    c, s = math.cos(theta), math.sin(theta)
    
    # Common Matrix Calculation Helper
    def get_affine_params(tex_transform, tex_w, tex_h, m_per_px_tex):
        # Scale
        scale = m_per_px_out / m_per_px_tex
        
        # Matrix M = S * R
        # [[Sc -Ss], [Ss Sc]] ? No, Inverse mapping.
        # We want Input = M * Output + Offset
        # Input (x,y) is Texture Px. Output is Screen Px.
        
        # M maps Output (rotated) -> Input (aligned)
        # R_inv maps Output -> Unrotated Frame
        # S_inv maps Unrotated Frame -> Texture Px
        
        # R (Output -> GeoDelta) = [[cos, sin], [-sin, cos]] * scale
        # Actually it's easier to build the matrix:
        # x_src = m00*y_out + m01*x_out + off_x
        # y_src = m10*y_out + m11*x_out + off_y
        
        m00 = scale * c
        m01 = scale * s
        m10 = -scale * s
        m11 = scale * c
        
        matrix = cp.array([[m00, m01], [m10, m11]])
        
        # Offset
        # Center of Output (h/2, w/2) maps to Center of Input (cx, cy)
        # cx, cy = Input pixel coord of (center_e, center_n)
        
        if tex_transform:
            # Ortho
            # ~tex_transform * (e, n)
            inv = ~tex_transform
            cx, cy = inv * (center_e, center_n)
        else:
            # WMS (Global Pixel space reference)
            # Need to handle WMS offset per function call
            cx, cy = 0, 0 

        # offset = center_in - M @ center_out
        oc_y, oc_x = height/2.0, width/2.0
        
        moy = m00 * oc_y + m01 * oc_x
        mox = m10 * oc_y + m11 * oc_x
        
        off_y = cy - moy
        off_x = cx - mox
        
        return matrix, [off_y, off_x]

    # --- Render Layers ---
    layers = []
    
    # 1. Ortho & Vectors (Share Transform)
    if _CONTEXT.is_ready and _CONTEXT.ortho_texture is not None:
        # m per px of ortho
        m_px = abs(_CONTEXT.ortho_transform.a)
        mtx, off = get_affine_params(_CONTEXT.ortho_transform, 0, 0, m_px)
        
        # Sample Ortho
        ortho_layer = _affine_sample(_CONTEXT.ortho_texture, mtx, off, height, width)
        layers.append(ortho_layer)
        
        # Sample Vectors (Exact same matrix/offset)
        if _CONTEXT.has_vectors and _CONTEXT.vector_texture is not None:
             vec_layer = _affine_sample(_CONTEXT.vector_texture, mtx, off, height, width)
             # Blend Vectors atop Ortho immediately? Or add to stack? 
             # Let's blend immediately to save memory
             # alpha blend
             alpha = vec_layer[:,:,3:4].astype(cp.float32) / 255.0
             ortho_layer[:,:,:3] = vec_layer[:,:,:3] * alpha + ortho_layer[:,:,:3] * (1.0 - alpha)
             # Max alpha? No, keep ortho alpha (usually 255)
             
    else:
        # Fallback empty
        layers.append(cp.zeros((height, width, 4), dtype=cp.uint8))

    # 2. WMS
    if _CONTEXT.is_ready and _CONTEXT.wms_texture is not None:
        # Calculate WMS specific params
        # Center Lat/Lon
        clon, clat = transform(dataset.crs, "EPSG:4326", [center_e], [center_n])
        clon, clat = clon[0], clat[0]
        
        # WMS Resolution at this Lat
        res_wms = (math.cos(math.radians(clat)) * 2 * math.pi * 6378137) / (256 * 2**_CONTEXT.wms_zoom)
        
        # Center Pixel in WMS texture
        # Global px
        gpx, gpy = _latlon_to_pixel(_clamp_latlon(clat, -85, 85), clon, _CONTEXT.wms_zoom)
        # Local px
        wms_ox, wms_oy = _CONTEXT.wms_bounds_px
        cx, cy = gpx - wms_ox, gpy - wms_oy
        
        # Matrix
        scale = m_per_px_out / res_wms
        m00 = scale * c
        m01 = scale * s
        m10 = -scale * s
        m11 = scale * c
        matrix = cp.array([[m00, m01], [m10, m11]])
        
        oc_y, oc_x = height/2.0, width/2.0
        moy = m00 * oc_y + m01 * oc_x
        mox = m10 * oc_y + m11 * oc_x
        
        off_y = cy - moy
        off_x = cx - mox
        
        wms_layer = _affine_sample(_CONTEXT.wms_texture, matrix, [off_y, off_x], height, width)
        
        # Blend (WMS is background)
        # Final = Ortho over WMS
        top = layers[0]
        bot = wms_layer
        
        alpha = top[:,:,3:4].astype(cp.float32) / 255.0
        out_rgb = top[:,:,:3].astype(cp.float32) * alpha + bot[:,:,:3].astype(cp.float32) * (1.0 - alpha)
        
        final_gpu = cp.dstack((out_rgb, cp.full((height, width, 1), 255, dtype=cp.float32))).astype(cp.uint8)
        
    else:
        final_gpu = layers[0]

    # --- 3. Icons / HUD (GPU Blit) ---
    # We blit the arrow icon in the center (static position, rotated icon)
    # Actually, the icon is always center screen facing UP relative to map flow?
    # No, standard is: Map rotates, Icon stays UP? Or Map Fixed, Icon Rotates?
    # The code rotates the map by -heading. So the map is "Heading Up".
    # Therefore, the Icon should point UP (0 deg rotation relative to screen).
    # Wait, existing code passes `heading` to map rotation. 
    # Validated: If map rotates by -heading, the "Forward" direction is UP. 
    # So the icon should be drawn pointing UP.
    
    # Draw simple triangle/arrow on GPU directly?
    # Or blit the texture we made in preload.
    # If we blit, we just copy it to center.
    if _CONTEXT.icon_texture is not None:
        # Blit center
        ih, iw, _ = _CONTEXT.icon_texture.shape
        cx, cy = width//2, height//2
        # Simple overlay
        # Alpha blend
        ic = _CONTEXT.icon_texture
        start_y = cy - ih//2
        start_x = cx - iw//2
        
        # simplistic bound check
        if start_y >= 0 and start_x >= 0 and start_y+ih < height and start_x+iw < width:
             patch_bg = final_gpu[start_y:start_y+ih, start_x:start_x+iw].astype(cp.float32)
             patch_ic = ic.astype(cp.float32)
             alpha = patch_ic[:,:,3:4] / 255.0 * icon_circle_opacity # Apply opacity
             
             # If icon texture is white, colorize? 
             # Assuming texture is pre-colored.
             out = patch_ic[:,:,:3] * alpha + patch_bg[:,:,:3] * (1.0 - alpha)
             final_gpu[start_y:start_y+ih, start_x:start_x+iw, :3] = out.astype(cp.uint8)

    # --- 4. Compass (CPU Fallback for now, or simple blit) ---
    # Compass is static in corner. Easy to blit if we had texture.
    # Leaving for CPU post-process to ensure quality text rendering.

    # --- Download ---
    final_arr = cp.asnumpy(final_gpu)
    img = Image.fromarray(final_arr, "RGBA")
    
    # Draw Compass (CPU is fine, it's just UI)
    if show_compass:
        c_pos = (width - compass_size_px - 10, compass_size_px + 10)
        # Note: Compass usually points North. Map is Heading Up.
        # So North is at -Heading degrees relative to Up.
        _draw_compass(img, c_pos, compass_size_px, -heading)

    return img

def preload_track_gpu(config: Any, jobs: List[Tuple]) -> None:
    """
    Wrapper to preload all resources into GPURenderContext.
    Called from app.py before the render loop.
    """
    if not HAS_GPU: return
    
    centers = [(j[1], j[2]) for j in jobs]
    
    # Open dataset momentarily to read data
    with rasterio.open(config.ortho_path) as dataset:
        # Load Vectors for baking
        vectors = load_vectors(
            dataset.crs,
            [layer.model_dump() for layer in config.vector_layers],
            config.vectors_paths,
            config.curves_path,
            config.line_color,
            config.line_width,
            config.boundary_color,
            config.boundary_width,
            config.point_color,
        )
        
        # Preload
        margin = config.map_half_width_m * 3.0 
        
        _CONTEXT.preload(
            dataset=dataset,
            center_points=centers,
            margin_m=margin,
            vectors=vectors,
            arrow_size=config.arrow_size_px,
            cone_len=config.cone_length_px,
            wms_source=config.wms_source
        )

