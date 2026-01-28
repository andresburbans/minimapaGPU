
import math
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image, ImageDraw

# Try importing GPU libraries (CuPy)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("[GPU] CuPy not found, falling back to CPU logic implicitly or failing.")

# Import constants and draw helpers from CPU render module
from render import (
    Segment,
    _to_rgb,
    _normalize,
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

# Constants
_ROTATION_OVERSCAN = 1.5

def init_gpu():
    """Check and return GPU status."""
    if not HAS_GPU:
        return {"available": False, "backend": "cpu"}
    try:
        # Trigger initialization
        cnt = cp.cuda.runtime.getDeviceCount()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get('name', b'Unknown').decode('utf-8')
        return {"available": True, "count": cnt, "backend": "cupy", "device": name}
    except Exception as e:
        return {"available": False, "error": str(e)}

def _normalize_gpu_rgba(arr_gpu, nodata=None):
    """
    Normalize raster data on GPU and handle transparency.
    Args:
        arr_gpu: (C, H, W) array on GPU.
        nodata: Value to treat as transparent.
    Returns:
        (H, W, 4) uint8 array on GPU (RGBA).
    """
    c, h, w = arr_gpu.shape
    
    # Identify alpha/mask
    if c == 4:
        # Already has alpha
        rgb = arr_gpu[:3]
        alpha = arr_gpu[3]
    else:
        # Create alpha from nodata
        rgb = arr_gpu[:3] if c >= 3 else cp.repeat(arr_gpu[0][None, ...], 3, axis=0)
        if nodata is not None:
            # Check for nodata equality (handling float nans if needed)
            if cp.isnan(float(nodata)):
                 mask = ~cp.isnan(arr_gpu[0])
            else:
                 mask = arr_gpu[0] != nodata
            alpha = mask.astype(cp.uint8) * 255
        else:
            alpha = cp.full((h, w), 255, dtype=cp.uint8)

    # Normalize RGB (Robust Min-Max)
    # We want to normalize based on valid pixels only
    if rgb.dtype != cp.uint8:
        rgb_f = rgb.astype(cp.float32)
        
        # Flatten valid pixels for stats
        # We use a stride or subsample for speed if the image is huge?
        # For max quality, full check.
        # But we can verify if just min/max is enough.
        # Let's use robust percentiles if possible? CuPy has percentile.
        # But simple min/max is faster.
        
        # Using simple min/max for speed generally suffices for visual preview
        # unless there are crazy outliers.
        
        valid_mask = alpha > 0
        if valid_mask.any():
            # Apply mask to all 3 channels? No, mask is 2D.
            # We can compute min/max per channel or global?
            # Global min/max preserves color balance.
            
            # Mask expansion for 3 channels
            # valid_mask_3d = cp.broadcast_to(valid_mask[None, :, :], rgb_f.shape)
            # vals = rgb_f[valid_mask_3d]
            # min_val = cp.nanmin(vals)
            # max_val = cp.nanmax(vals)
            
            # Faster: Min/Max of the whole array (ignoring nodata if we replace it first)
            # Replace invalid with NaN temporarily?
            # Or just calc min/max.
            
            min_val = cp.nanmin(rgb_f)
            max_val = cp.nanmax(rgb_f)
            
            if max_val - min_val < 1e-5:
                rgb_u8 = cp.zeros_like(rgb, dtype=cp.uint8)
            else:
                scaled = (rgb_f - min_val) / (max_val - min_val)
                rgb_u8 = (scaled * 255).clip(0, 255).astype(cp.uint8)
        else:
            rgb_u8 = cp.zeros(rgb.shape, dtype=cp.uint8)
    else:
        rgb_u8 = rgb

    # Stack to (H, W, 4)
    # rgb_u8 is (3, H, W), alpha is (H, W)
    
    r = rgb_u8[0]
    g = rgb_u8[1]
    b = rgb_u8[2]
    
    # Stack depth-wise
    return cp.dstack((r, g, b, alpha))

def _rotate_and_crop_gpu(img_gpu, angle, out_w, out_h):
    """
    Rotate and crop on GPU.
    img_gpu: (H, W, 4)
    angle: rotation in degrees (CCW).
    out_w, out_h: output dimensions.
    """
    # Pad to avoid cropping during rotation?
    # ndimage.rotate reshapes by default.
    # We want to rotate around center.
    
    # If the input is the "supersampled square", its center is the target center.
    # We can use reshape=False to keep the bounding box if it's large enough,
    # OR reshape=True and then crop center.
    # Since we sized the read window to cover the rotated view (overscan),
    # reshape=False works if the box is a square covering the diagonal.
    # But reshape=True is safer to not lose corners, then we crop.
    
    rotated = ndimage.rotate(img_gpu, angle, axes=(1, 0), reshape=False, order=1, prefilter=False)
    # axes=(1,0) -> (y, x)
    
    rh, rw, rc = rotated.shape
    start_y = (rh - out_h) // 2
    start_x = (rw - out_w) // 2
    
    # Safe crop
    # If start_x < 0, we need padding? This shouldn't happen with correct 'overscan'.
    if start_x < 0: start_x = 0
    if start_y < 0: start_y = 0
    
    end_x = min(start_x + out_w, rw)
    end_y = min(start_y + out_h, rh)
    
    # If smaller than requested, we might need padding (unlikely if logic is right)
    cropped = rotated[start_y:end_y, start_x:end_x, :]
    
    return cropped

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
) -> Image.Image:
    
    if not HAS_GPU:
        raise RuntimeError("GPU acceleration requested but CuPy is not available.")

    # === 1. Setup Geometry ===
    ss_factor = 2.0 
    meters_per_pixel = (map_half_width_m * 2.0) / width
    
    # View diagonal in pixels
    diag_px = math.sqrt(width**2 + height**2)
    # Margin for rotation safety
    render_size_px = int(diag_px * 1.15)
    
    # Size in meters to read
    render_size_m = render_size_px * meters_per_pixel
    
    # Bounds to read
    xmin = center_e - render_size_m / 2
    xmax = center_e + render_size_m / 2
    ymin = center_n - render_size_m / 2
    ymax = center_n + render_size_m / 2
    
    # Read Resolution (Supersampled)
    ss_render_size_px = int(render_size_px * ss_factor)
    
    # === 2. Read Raster (CPU bottleneck) ===
    # Using windowed read
    window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
    
    # Read directly as numpy array
    # boundless=True fills with nodata outside bounds
    cpu_data = dataset.read(
        window=window,
        out_shape=(dataset.count, ss_render_size_px, ss_render_size_px),
        resampling=Resampling.bilinear,
        boundless=True,
        fill_value=dataset.nodata
    )
    
    # === 3. Upload and Process on GPU ===
    # Normalize and Convert to RGBA
    gpu_data = cp.asarray(cpu_data)
    gpu_rgba = _normalize_gpu_rgba(gpu_data, nodata=dataset.nodata)
    
    # Resize (Downsample to Render Size) - Anti-aliasing step
    # gpu_rgba is (SS_H, SS_W, 4)
    # We want (H, W, 4) where H = render_size_px
    zoom = 1.0 / ss_factor
    
    # Use order=1 (bilinear) for speed, enough for downsampling
    # Applying zoom to spatial dims (0, 1) only, keeping channels (2)
    gpu_resized = ndimage.zoom(gpu_rgba, (zoom, zoom, 1), order=1)
    
    # === 4. Rotate on GPU ===
    # Heading points UP. Image logic:
    # We rotate by 'heading' (which is the vehicle's azimuth).
    # If heading is 0, no rotation.
    # If heading is 90 (East), we want East to be Up?
    # Actually, if we travel East, the map should move Left?
    # Standard: "Heading Up" mode. 
    # Rotate image by 'heading'.
    gpu_rotated = ndimage.rotate(gpu_resized, heading, axes=(1, 0), reshape=False, order=1, prefilter=False, mode='constant', cval=0)
    
    # === 5. Crop Center on GPU ===
    # We have a rotated square. Now we crop the view window (width x height)
    rh, rw, _ = gpu_rotated.shape
    start_y = (rh - height) // 2
    start_x = (rw - width) // 2
    
    # Pad if necessary (unlikely given margins)
    # Simple slicing
    if start_x < 0: start_x = 0
    if start_y < 0: start_y = 0
    end_x = start_x + width
    end_y = start_y + height
    
    # Check bounds
    end_x = min(end_x, rw)
    end_y = min(end_y, rh)
    
    gpu_final_map = gpu_rotated[start_y:end_y, start_x:end_x]
    
    # === 6. WMS Background Integration (Hybrid) ===
    # We calculate the WMS bounds.
    # For optimal quality, we should fetch WMS, rotate it, and composite.
    # To save time, we can reuse logic or implement a simplified GPU compositor.
    
    # Let's download the map first. Blending heavily on GPU requires the WMS to be on GPU.
    # Fetching WMS is CPU bound (network).
    
    # Fetch WMS (CPU)
    clip_margin = map_half_width_m * 2.0
    wms_bbox_geo = (
        center_e - clip_margin,
        center_n - clip_margin,
        center_e + clip_margin,
        center_n + clip_margin,
    )
    
    # We use the cached fetcher
    # Use zoom=19 or calculated suitable zoom?
    # _fetch_wms_mosaic_for_bounds logic is inside render.py, uses fixed logic or passed zoom?
    # It takes (w, s, e, n, zoom).
    # We need a zoom level that matches resolution.
    # meters_per_pixel = 0.5 (approx).
    # Zoom 19 is ~0.3m/px. Zoom 18 ~0.6m/px.
    zoom_level = 19
    
    # Transform to latlon for WMS
    w_geo, s_geo, e_geo, n_geo = transform_bounds(
        dataset.crs, "EPSG:4326", 
        wms_bbox_geo[0], wms_bbox_geo[1], wms_bbox_geo[2], wms_bbox_geo[3]
    )
    
    wms_img, (wms_left, wms_top) = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, zoom_level)
    
    # We need to rotate and crop the WMS to match the frame.
    # Let's do this on GPU to be cool and fast (after fetch).
    wms_arr = cp.asarray(np.array(wms_img.convert("RGBA"))) # (H_wms, W_wms, 4)
    
    # Align WMS to view center
    # Center of view in LatLon:
    center_lon, center_lat = transform_bounds(dataset.crs, "EPSG:4326", center_e, center_n, center_e, center_n)
    # This might return lists? No, unzip.
    center_lon = center_lon[0]
    center_lat = center_lat[0]
    
    cx_glob, cy_glob = _latlon_to_pixel(center_lat, center_lon, zoom_level)
    
    rel_cx = cx_glob - wms_left
    rel_cy = cy_glob - wms_top
    
    # We extracting a crop around center that matches our "render_size_px" (the diagonal one)
    # BUT WMS resolution might differ from 'render_size_px'.
    # We need to scale WMS to match 'render_size_px'.
    
    # Instead, let's just use the robust PIL fallback for WMS if GPU logic is too complex for alignment right now?
    # NO, use GPU.
    
    # Crop WMS to a square around center
    wms_crop_size = min(wms_arr.shape[0], wms_arr.shape[1]) # Simplify or use safe crop
    # We want a crop covering the view.
    # Let's trust the logic: extract a square of size 'render_size_px' * (res_ratio)?
    
    # Simpler: Upload WMS, Rotate, Scale to (width, height)?
    # We can follow the same pattern: Crop diagonal -> Rotate -> Crop Final.
    
    # Center crop WMS
    # Size needed: diag of view.
    # Assuming WMS fetch was big enough (clip_margin used logic).
    
    # Just center crop the WMS array to match the 'spatial coverage' of the Map array?
    # They have different resolutions.
    # Map: 'meters_per_pixel'. WMS: 'wms_meters_per_px'.
    # We need to scale WMS to match Map Resolution.
    # OR, better: Scale Map to Screen. Scale WMS to Screen. Composite.
    # We already Scaled Map to Screen (gpu_rotated is Screen Resolution).
    
    # We need to Rotate WMS by 'heading' and Scale it to match 'width x height'.
    # Since we don't have exact pixel logic easily on GPU without homography...
    # Let's revert WMS composite to CPU PIL for safety, BUT using the GPU-rotated map download.
    # GPU Rotated Map (gpu_final_map) is "Perfect".
    # We download it.
    
    final_map_cpu = cp.asnumpy(gpu_final_map).astype(np.uint8)
    if final_map_cpu.ndim == 2: # No alpha?
         # Handle case
         pass
         
    map_img = Image.fromarray(final_map_cpu, mode="RGBA")
    
    # --- WMS Composite (CPU) ---
    # We use the helper from render_gpu (renaming/fixing it) or define here.
    # We need to rotate WMS.
    # crop wms around center
    
    # Using PIL for WMS rotation is standard.
    # We just need to make sure we align it.
    # The existing 'compose_wms_background' in render_gpu.py was trying to do this.
    
    # Reuse logical adaptation:
    crop_size = max(width, height) * 1.5 # Margin for rotation
    crop_l = int(rel_cx - crop_size / 2)
    crop_t = int(rel_cy - crop_size / 2)
    crop_r = int(crop_l + crop_size)
    crop_b = int(crop_t + crop_size)
    
    wms_crop = wms_img.crop((crop_l, crop_t, crop_r, crop_b))
    wms_rotated = _rotate_image(wms_crop, heading if heading else 0, (width, height))
    
    # Composite: Map over WMS
    # Map might be smaller if we hit edges of raster?
    # map_img alpha channel determines transparency.
    
    if map_img.size != wms_rotated.size:
        map_img = map_img.resize(wms_rotated.size)
        
    final_composite = Image.alpha_composite(wms_rotated.convert("RGBA"), map_img)
    
    # === 7. Draw Vectors (CPU Post-Process) ===
    # Coordinate transformation function
    # Map coords (e, n) -> Screen pixels (x, y)
    # Steps:
    # 1. Translate to Center (e - center_e, n - center_n) (meters)
    # 2. Rotate by -heading (radians)
    # 3. Scale by 1/meters_per_pixel (pixels)
    # 4. Translate to Screen Center (width/2, height/2)
    # 5. Y-flip? 
    #    Map Y is North (Up).
    #    Screen Y is Down.
    #    So after rotation (where +Y is "Forward/Up"), we convert +Ym to -Ypx.
    
    cx_screen = width / 2
    cy_screen = height / 2
    
    # We rotated the Map by 'heading' (CCW).
    # We must rotate the vectors by the same amount.
    angle_rad = math.radians(heading)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    scale = 1.0 / meters_per_pixel
    
    def transform_point(e, n):
        # 1. Translate
        dx = e - center_e
        dy = n - center_n
        
        # 2. Rotate
        # Standard rotation: x' = x cos - y sin, y' = x sin + y cos
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        # 3. Scale & Flip Y & Offset
        sx = cx_screen + rx * scale
        sy = cy_screen - ry * scale # Y flip for screen
        
        return sx, sy

    draw = ImageDraw.Draw(final_composite, "RGBA")
    
    for geom_iter, color, line_width, pattern in vectors:
        for geom in geom_iter:
             _draw_geometry_precise(draw, geom, transform_point, color, line_width, pattern)
             
    # === 8. HUD Elements ===
    center_px = (width // 2, height // 2)
    _draw_cone(final_composite, center_px, 0.0, cone_angle_deg, cone_length_px, cone_opacity)
    _draw_center_icon(
        final_composite,
        center_px,
        arrow_size_px,
        icon_circle_opacity,
        icon_circle_size_px,
        0.0 # Icon always points UP
    )
    
    if show_compass:
        compass_size = max(15, compass_size_px)
        compass_margin = max(10, compass_size // 2)
        compass_pos = (width - compass_margin - compass_size, compass_margin + compass_size)
        # Compass North rotates with map?
        # Map is rotated by Heading. North is at -heading.
        _draw_compass(final_composite, compass_pos, compass_size, -heading)
        
    return final_composite
