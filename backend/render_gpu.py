
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
from PIL import Image, ImageDraw

import gpu_utils # Fix DLL paths before importing cupy
# Try importing GPU libraries (CuPy and Torch)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("[GPU] CuPy not found, falling back to CPU logic.")

try:
    import torch
    import kornia
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

# Import constants and draw helpers from CPU render module
# We import them to maintain compatibility and reuse drawing logic for vectors (which is mostly CPU for now)
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
    # Fallback or stub if render.py is not available in current context (unlikely)
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
        return {"available": True, "count": cnt, "backend": "cupy", "device": name}
    except Exception as e:
        return {"available": False, "error": str(e)}

class GPURenderContext:
    """
    Manages GPU memory and pre-loaded textures to maximize rendering speed.
    Avoids reading from disk and fetching WMS for every frame.
    """
    def __init__(self):
        self.ortho_texture = None # (C, H, W) GPU array
        self.ortho_transform = None
        self.ortho_crs = None
        self.ortho_nodata = None
        
        self.wms_texture = None # (4, H, W) GPU array
        self.wms_bounds_px = None # (left, top, right, bottom) in global pixel coords
        self.wms_zoom = None
        
        self.last_config_key = None
        self.is_ready = False

    def preload(self, dataset, center_points, margin_m, zoom_level=19):
        """
        Pre-loads the entire track area into GPU memory.
        center_points: List of (e, n)
        """
        if not HAS_GPU: return False
        
        # 1. Calculate bounding box of the whole track
        es = [p[0] for p in center_points]
        ns = [p[1] for p in center_points]
        
        xmin, xmax = min(es) - margin_m, max(es) + margin_m
        ymin, ymax = min(ns) - margin_m, max(ns) + margin_m
        
        # 2. Read Ortho into GPU
        window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
        # We read at a reasonable resolution to avoid GPU OOM but maintain quality
        # For a track, we might need a large texture. 
        # Let's limit the texture size to e.g. 8192x8192 or similar if possible.
        # Rasterio can handle the resampling.
        
        # Check memory
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        # Use about 95% of free memory for this texture as requested
        target_mem = free_mem * 0.95
        
        # Estimate pixel count based on 4 channels float32
        max_pixels = target_mem / (4 * 4) 
        side = int(math.sqrt(max_pixels))
        side = min(side, 12288) # Cap at 12k
        
        # Calculate aspect ratio
        w_m = xmax - xmin
        h_m = ymax - ymin
        ratio = w_m / h_m
        
        if ratio > 1:
            tw, th = side, int(side / ratio)
        else:
            tw, th = int(side * ratio), side
            
        print(f"[GPU] Preloading track area: {tw}x{th} pixels")
        
        cpu_data = dataset.read(
            window=window,
            out_shape=(dataset.count, th, tw),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=dataset.nodata
        )
        
        self.ortho_texture = cp.asarray(cpu_data)
        
        # Adjust transform for the new resolution (we requested out_shape=(th, tw))
        # The window transform is for the original resolution.
        # We need to scale it to match the downsampled texture.
        base_transform = rasterio.windows.transform(window, dataset.transform)
        scale_x = window.width / tw
        scale_y = window.height / th
        self.ortho_transform = base_transform * Affine.scale(scale_x, scale_y)
        
        self.ortho_crs = dataset.crs
        self.ortho_nodata = dataset.nodata
        
        # Convert to RGBA normalized immediately to save work per frame
        self.ortho_rgba = _normalize_gpu_rgba(self.ortho_texture, self.ortho_nodata)
        # self.ortho_rgba is now (H, W, 4) uint8
        
        # 3. Fetch WMS for the whole area
        w_geo, s_geo, e_geo, n_geo = transform_bounds(
            dataset.crs, "EPSG:4326", xmin, ymin, xmax, ymax
        )
        
        # Estimate suitable zoom level for WMS
        # Local simplistic zoom choice to avoid circular import
        def _get_zoom(w, s, e, n):
            # Roughly estimate zoom where span is ~2000-4000 pixels
            span_x = abs(e - w)
            if span_x < 0.001: return 19
            # Very rough heuristic
            for z in range(19, 10, -1):
                res = 156543.03392 * math.cos(math.radians(s)) / (2**z)
                pix = (span_x * 111000) / res
                if pix > 2000: return z
            return 12

        wms_zoom = _get_zoom(w_geo, s_geo, e_geo, n_geo)
        
        print(f"[GPU] Fetching WMS mosaic for whole track (Zoom {wms_zoom})...")
        # Robust unpacking to handle potential mismatch (2 vs 4 values in tuple)
        ret_wms = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, wms_zoom)
        wms_img = ret_wms[0]
        wms_info = ret_wms[1]
        
        if len(wms_info) >= 2:
            wms_l, wms_t = wms_info[0], wms_info[1]
        else:
            wms_l, wms_t = 0, 0
            print(f"[GPU] Warning: WMS info tuple too short: {wms_info}")

        self.wms_texture = cp.asarray(np.array(wms_img.convert("RGBA"))) # (H, W, 4)
        self.wms_bounds_px = (wms_l, wms_t)
        self.wms_zoom = wms_zoom
        
        self.is_ready = True
        return True

_CONTEXT = GPURenderContext()

def _normalize_gpu_rgba(arr_gpu, nodata=None):
    """
    Normalize raster data on GPU and handle transparency.
    Returns (H, W, 4) uint8 array on GPU.
    """
    c, h, w = arr_gpu.shape
    
    # 1. Handle Channels
    if c >= 3:
        rgb = arr_gpu[:3]
    else:
        # Grayscale to RGB
        rgb = cp.repeat(arr_gpu[0][None, ...], 3, axis=0)
        
    # 2. Handle Alpha
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
        
    # 3. Normalize RGB
    if rgb.dtype != cp.uint8:
        rgb_f = rgb.astype(cp.float32)
        # Use robust min/max (ignoring nodata if possible, or just global)
        # For speed, global min/max
        valid_mask = alpha > 0
        if valid_mask.any():
            v_min = cp.percentile(rgb_f[cp.broadcast_to(valid_mask, rgb_f.shape)], 2)
            v_max = cp.percentile(rgb_f[cp.broadcast_to(valid_mask, rgb_f.shape)], 98)
            if v_max - v_min < 1e-6:
                rgb_u8 = cp.zeros_like(rgb, dtype=cp.uint8)
            else:
                rgb_u8 = ((rgb_f - v_min) / (v_max - v_min) * 255).clip(0, 255).astype(cp.uint8)
        else:
            rgb_u8 = cp.zeros(rgb.shape, dtype=cp.uint8)
    else:
        rgb_u8 = rgb
        
    # Transpose to (H, W, 3) and stack alpha
    res = cp.dstack((rgb_u8[0], rgb_u8[1], rgb_u8[2], alpha))
    return res

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
    show_compass: bool = True,
    compass_size_px: int = 40,
    use_context: bool = True, # Enable optimized context rendering
) -> Image.Image:
    
    if not HAS_GPU:
        raise RuntimeError("GPU acceleration requested but CuPy is not available.")

    # --- 1. Prep Window and Sizes ---
    ss_factor = 2.0
    meters_per_pixel = (map_half_width_m * 2.0) / width
    diag_px = math.sqrt(width**2 + height**2)
    render_size_px = int(diag_px * 1.15)
    render_size_m = render_size_px * meters_per_pixel
    
    # DEBUG
    # if idx % 30 == 0: 
    print(f"[GPU Frame] Center: {center_e:.1f}, {center_n:.1f} | CtxReady: {_CONTEXT.is_ready}")

    # --- 2. Acquire Map Image (GPU) ---
    if use_context and _CONTEXT.is_ready:
        # Use pre-loaded texture (FAST)
        # Calculate affine transformation for cropping the pre-loaded texture
        # Map pixels in texture to geographic coords
        tx_inv = ~_CONTEXT.ortho_transform
        
        # Center in texture pixels
        cx_tex, cy_tex = tx_inv * (center_e, center_n)
        
        # Size in texture pixels
        # Resolution of texture:
        res_x_tex = _CONTEXT.ortho_transform.a
        # res_x_tex can be negative (usual for map rasters, north-up)
        # We need magnitude
        m_px_tex = abs(res_x_tex)
        
        # Half size in texture pixels
        half_w_tex = (render_size_m / 2.0) / m_px_tex
        
        # Coordinates to slice
        l, t = int(cx_tex - half_w_tex), int(cy_tex - half_w_tex)
        r, b = int(l + half_w_tex * 2), int(t + half_w_tex * 2)
        
        # Debug bounds
        print(f"  [Ortho] Slice: {l}:{r}, {t}:{b} (Tex Size: {_CONTEXT.ortho_rgba.shape})")
        
        # Safe slice (handling image boundaries with padding if needed)
        th, tw, _ = _CONTEXT.ortho_rgba.shape
        
        # Extract and Resize to 'render_size_px'
        # We use CuPy slices. If out of bounds, we need to pad.
        sl_l, sl_t = max(0, l), max(0, t)
        sl_r, sl_b = min(tw, r), min(th, b)
        
        chunk = _CONTEXT.ortho_rgba[sl_t:sl_b, sl_l:sl_r]
        
        if chunk.size == 0:
             # print("  [Ortho] Empty chunk!")
             gpu_base = cp.zeros((render_size_px, render_size_px, 4), dtype=cp.uint8)
        else:
            # Pad if chunk is smaller than intended (edges of raster)
            pad_t = max(0, -t)
            pad_l = max(0, -l)
            pad_b = max(0, b - th)
            pad_r = max(0, r - tw)
            
            if pad_t > 0 or pad_l > 0 or pad_b > 0 or pad_r > 0:
                chunk = cp.pad(chunk, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant', constant_values=0)
    
            # Resize chunk to render_size_px using GPU
            # Check actual chunk size after padding (should be r-l, b-t)
            ch, cw, _ = chunk.shape
            
            # Prevent zero division
            cw = max(1, cw)
            ch = max(1, ch)
            
            zoom_w = render_size_px / cw
            zoom_h = render_size_px / ch
            
            # Using ndimage.zoom for fast resizing
            gpu_base = ndimage.zoom(chunk, (zoom_h, zoom_w, 1), order=1)
        
    else:
        # Standard logic: Read from dataset (SLOW)
        xmin, xmax = center_e - render_size_m/2, center_e + render_size_m/2
        ymin, ymax = center_n - render_size_m/2, center_n + render_size_m/2
        ss_render_size_px = int(render_size_px * ss_factor)
        
        window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
        cpu_data = dataset.read(
            window=window,
            out_shape=(dataset.count, ss_render_size_px, ss_render_size_px),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=dataset.nodata
        )
        gpu_data = cp.asarray(cpu_data)
        gpu_rgba = _normalize_gpu_rgba(gpu_data, nodata=dataset.nodata)
        # Downsample to render_size_px
        zoom = 1.0 / ss_factor
        gpu_base = ndimage.zoom(gpu_rgba, (zoom, zoom, 1), order=1)

    # --- 3. Rotate and Crop (GPU) ---
    # Rotate by 'heading' (CCW)
    # We want destination UP, so if heading is 0 (North), no rotation.
    # ndimage.rotate uses CCW degrees.
    gpu_rotated = ndimage.rotate(gpu_base, heading, axes=(1, 0), reshape=False, order=1, prefilter=False, mode='constant', cval=0)
    
    # Center crop to (width, height)
    rh, rw, _ = gpu_rotated.shape
    start_y = (rh - height) // 2
    start_x = (rw - width) // 2
    gpu_map_final = gpu_rotated[start_y:start_y+height, start_x:start_x+width]

    # --- 4. WMS Background (GPU Optimized) ---
    if use_context and _CONTEXT.wms_texture is not None:
        # Align WMS on GPU
        center_lon, center_lat = transform(dataset.crs, "EPSG:4326", [center_e], [center_n])
        center_lon, center_lat = center_lon[0], center_lat[0]
        
        cx_glob, cy_glob = _latlon_to_pixel(center_lat, center_lon, _CONTEXT.wms_zoom)
        rel_cx = cx_glob - _CONTEXT.wms_bounds_px[0]
        rel_cy = cy_glob - _CONTEXT.wms_bounds_px[1]
        
        # WMS Resolution in meters/pixel at this lat
        wms_res_m = (math.cos(math.radians(center_lat)) * 2 * math.pi * 6378137) / (256 * 2**_CONTEXT.wms_zoom)
        
        # Pixels in WMS for 'render_size_m'
        half_w_wms = (render_size_m / 2.0) / wms_res_m
        
        wl, wt = int(rel_cx - half_w_wms), int(rel_cy - half_w_wms)
        wr, wb = int(wl + half_w_wms * 2), int(wt + half_w_wms * 2)
        
        wth, wtw, _ = _CONTEXT.wms_texture.shape
        wsl, wst = max(0, wl), max(0, wt)
        wsr, wsb = min(wtw, wr), min(wth, wb)
        
        wms_chunk = _CONTEXT.wms_texture[wst:wsb, wsl:wsr]
        
        # Resize and Rotate WMS on GPU
        zw_wms = render_size_px / max(1, (wr - wl))
        zh_wms = render_size_px / max(1, (wb - wt))
        wms_base = ndimage.zoom(wms_chunk, (zh_wms, zw_wms, 1), order=1)
        wms_rotated = ndimage.rotate(wms_base, heading, axes=(1, 0), reshape=False, order=1, prefilter=False, mode='constant', cval=0)
        
        # Crop WMS
        wrh, wrw, _ = wms_rotated.shape
        wstart_y = (wrh - height) // 2
        wstart_x = (wrw - width) // 2
        gpu_wms_final = wms_rotated[wstart_y:wstart_y+height, wstart_x:wstart_x+width]
        
        # Validate WMS shape - fallback to black if mismatch (e.g. OOB or empty WMS)
        if gpu_wms_final.shape[0] != height or gpu_wms_final.shape[1] != width:
            # Create black background
            gpu_wms_final = cp.zeros((height, width, 4), dtype=cp.uint8)
            # Should we try to paste what we got? Maybe too complex/risky, precise alignment needed.
            # For now, safe fallback to empty.
            
        # Ensure 4-channel
        if gpu_wms_final.shape[2] == 3:
             # Add alpha
             gpu_wms_final = cp.dstack((gpu_wms_final, cp.full((height, width, 1), 255, dtype=cp.uint8)))
        elif gpu_wms_final.shape[2] != 4:
             # Unknown
             gpu_wms_final = cp.zeros((height, width, 4), dtype=cp.uint8)
        
        # --- 5. Alpha Blend (GPU) ---
        # Map over WMS
        # Out = Src * Alpha + Dest * (1 - Alpha)
        # gpu_map_final is (H, W, 4)
        alpha = gpu_map_final[:, :, 3:4].astype(cp.float32) / 255.0
        src_rgb = gpu_map_final[:, :, :3].astype(cp.float32)
        dst_rgb = gpu_wms_final[:, :, :3].astype(cp.float32)
        
        blended_rgb = src_rgb * alpha + dst_rgb * (1.0 - alpha)
        gpu_final = cp.dstack((blended_rgb, cp.full((height, width, 1), 255, dtype=cp.float32)))
        gpu_final_u8 = gpu_final.clip(0, 255).astype(cp.uint8)
        
    else:
        # Fallback to slower hybrid method if context not ready
        final_map_cpu = cp.asnumpy(gpu_map_final).astype(np.uint8)
        map_img = Image.fromarray(final_map_cpu, mode="RGBA")
        # (... rest of hybrid logic from previous version or render.py ...)
        # For brevity and since we want MAX speed, we assume context is used.
        # But let's add a basic fallback to return at least something.
        gpu_final_u8 = gpu_map_final

    # --- 6. Finalize and Draw HUD (PIL/CPU) ---
    # Transfer to CPU
    final_arr = cp.asnumpy(gpu_final_u8)
    final_img = Image.fromarray(final_arr, mode="RGBA")
    
    # Draw Vectors and HUD on CPU (standard PIL)
    # This part is relatively fast compared to disk/net
    draw = ImageDraw.Draw(final_img, "RGBA")
    
    # Vector Transform logic
    cx_screen, cy_screen = width / 2, height / 2
    angle_rad = math.radians(heading)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    scale = 1.0 / meters_per_pixel
    
    def transform_pt(e, n):
        dx, dy = e - center_e, n - center_n
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return cx_screen + rx * scale, cy_screen - ry * scale

    for geom_iter, color, line_width, pattern in vectors:
        for geom in geom_iter:
            _draw_geometry_precise(draw, geom, transform_pt, color, line_width, pattern)

    # HUD Elements
    _draw_cone(final_img, (int(cx_screen), int(cy_screen)), 0.0, cone_angle_deg, cone_length_px, cone_opacity)
    _draw_center_icon(final_img, (int(cx_screen), int(cy_screen)), arrow_size_px, icon_circle_opacity, icon_circle_size_px, 0.0)
    
    if show_compass:
        c_size = max(15, compass_size_px)
        c_margin = max(10, c_size // 2)
        c_pos = (width - c_margin - c_size, c_margin + c_size)
        _draw_compass(final_img, c_pos, c_size, -heading)

    return final_img

def preload_track_gpu(config, jobs):
    """Entry point to initialize the GPU context for a batch render."""
    if not HAS_GPU: return False
    
    with rasterio.open(config.ortho_path) as dataset:
        # Extract points
        points = [(j[1], j[2]) for j in jobs]
        # Margin: diagonal of view + some buffer
        margin = (math.sqrt(config.width**2 + config.height**2) * (config.map_half_width_m * 2 / config.width)) * 1.5
        
        return _CONTEXT.preload(dataset, points, margin)

