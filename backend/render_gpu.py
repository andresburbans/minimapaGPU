
import math
import logging
import io
import time
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform
from rasterio.transform import Affine
from PIL import Image, ImageDraw, ImageOps
from pyproj import CRS, Transformer

import gpu_utils 

# Try importing GPU libraries
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    from cupyx.scipy.ndimage import zoom
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("[GPU] CuPy not found, falling back to CPU logic.")

# Import helper functions from CPU render module
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
        _GLOBAL_NAV_ICON,
        _fetch_wms_mosaic_for_bounds,
        _latlon_to_pixel,
        _rotate_image,
        _clamp_latlon,
    )
except ImportError:
    print("[GPU] Warning: Could not import some functions from 'render.py'")

# --- GPU Utility Functions ---

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

def _alpha_composite_gpu(fg: cp.ndarray, bg: cp.ndarray) -> cp.ndarray:
    """
    Proper alpha compositing matching PIL.Image.alpha_composite using GPU.
    fg and bg are expected to be (H, W, 4) in uint8.
    """
    # Convert to float and normalize to 0-1
    fg_a = fg[:, :, 3:4].astype(cp.float32) / 255.0
    bg_a = bg[:, :, 3:4].astype(cp.float32) / 255.0
    
    # Calculate output alpha
    out_a = fg_a + bg_a * (1.0 - fg_a)
    
    # Avoid division by zero
    out_a_safe = cp.where(out_a > 1e-6, out_a, 1.0)
    
    # Calculate output RGB
    fg_rgb = fg[:, :, :3].astype(cp.float32)
    bg_rgb = bg[:, :, :3].astype(cp.float32)
    
    # Formula: (fg_rgb * fg_a + bg_rgb * bg_a * (1 - fg_a)) / out_a
    out_rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / out_a_safe
    
    # Convert back to uint8
    out_rgb = cp.clip(out_rgb, 0, 255).astype(cp.uint8)
    out_a_u8 = cp.clip(out_a * 255, 0, 255).astype(cp.uint8)
    
    return cp.dstack((out_rgb, out_a_u8))

def _gpu_downsample_box(arr: cp.ndarray, scale: int) -> cp.ndarray:
    """
    Fast box filter downsampling for integer scale.
    Expects (H, W, 4) in uint8.
    """
    if scale == 1:
        return arr
    
    if scale == 2:
         # Box filter 2x2
         arr_f = arr.astype(cp.float32)
         out = (arr_f[0::2, 0::2] + arr_f[1::2, 0::2] + arr_f[0::2, 1::2] + arr_f[1::2, 1::2]) / 4.0
         return out.astype(cp.uint8)
         
    # Fallback to linear zoom
    factor = 1.0 / scale
    return zoom(arr, (factor, factor, 1), order=1).astype(cp.uint8)

def _get_transformation_basis(heading: float, m_per_px: float):
    """
    Calculate basis vectors for the transformation.
    Matches render.py convention where Heading points UP.
    """
    rad = math.radians(heading)
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    
    # Heading points UP on screen.
    # Travel Vector T = (sin h, cos h) in East, North.
    # Vector Y (DOWN in image) points in OPPOSITE direction of travel: -T
    vec_y_e = m_per_px * (-sin_h)
    vec_y_n = m_per_px * (-cos_h)
    
    # Vector X (RIGHT in image) is 90 deg CW from Travel Vector T: (cos h, -sin h)
    vec_x_e = m_per_px * cos_h
    vec_x_n = m_per_px * (-sin_h)
    
    return vec_x_e, vec_x_n, vec_y_e, vec_y_n

def _sample_using_inverse_transform(
    texture_planar: cp.ndarray,
    center_e: float,
    center_n: float,
    heading: float,
    m_per_px_out: float,
    out_h: int,
    out_w: int,
    ortho_transform: Affine,
    mipmap_level: int = 0
) -> cp.ndarray:
    """
    Perform reverse projection sampling using GPU-accelerated affine transform.
    """
    # 1. Basis vectors
    vxe, vxn, vye, vyn = _get_transformation_basis(heading, m_per_px_out)
    
    # 2. Map basis to Texture Plane
    itf = ~ortho_transform
    level_scale = 1.0 / (2 ** mipmap_level)
    
    # Precise center in pixels
    cx_tex, cy_tex = itf * (center_e, center_n)
    cx_tex *= level_scale
    cy_tex *= level_scale
    
    # Derivatives for affine matrix (how many texture pixels per output pixel)
    d_col_dx = (itf.a * vxe + itf.b * vxn) * level_scale
    d_col_dy = (itf.a * vye + itf.b * vyn) * level_scale
    d_row_dx = (itf.d * vxe + itf.e * vxn) * level_scale
    d_row_dy = (itf.d * vye + itf.e * vyn) * level_scale

    # 3. Affine Matrix & Offset
    matrix = cp.array([
        [d_row_dy, d_row_dx],
        [d_col_dy, d_col_dx]
    ], dtype=cp.float32)
    
    off_r = cp.float32(cy_tex - (d_row_dy * out_h / 2.0 + d_row_dx * out_w / 2.0))
    off_c = cp.float32(cx_tex - (d_col_dy * out_h / 2.0 + d_col_dx * out_w / 2.0))
    offset = cp.array([off_r, off_c], dtype=cp.float32)

    # 4. Transform
    result_planar = cp.zeros((4, out_h, out_w), dtype=cp.uint8)
    for i in range(4):
        ndimage.affine_transform(
            texture_planar[i],
            matrix,
            offset=offset,
            output_shape=(out_h, out_w),
            output=result_planar[i],
            order=1,
            mode='constant',
            cval=0,
            prefilter=False
        )
        
    return cp.transpose(result_planar, (1, 2, 0))

def _sample_wms_layer_gpu_approx(
    wms_texture_planar: cp.ndarray,
    ortho_crs,
    center_e: float,
    center_n: float,
    heading: float,
    m_per_px_out: float,
    out_w: int, 
    out_h: int,
    wms_zoom: int,
    wms_bounds_px: Tuple[float, float]
) -> cp.ndarray:
    """
    Fast WMS sampling using EXACT basis logic for perfect stitching.
    """
    # 1. Basis vectors (same as ortho)
    vxe, vxn, vye, vyn = _get_transformation_basis(heading, m_per_px_out)
    
    # 2. Corners in Geo space for WMS projection
    # Corners in relative pixels
    corners_px = [
        (-out_w/2, -out_h/2), # TL
        (out_w/2, -out_h/2),  # TR
        (-out_w/2, out_h/2),  # BL
    ]
    
    geo_pts = []
    for px, py in corners_px:
        ge = center_e + px * vxe + py * vye
        gn = center_n + px * vxn + py * vyn
        geo_pts.append((ge, gn))
        
    e_pts = np.array([p[0] for p in geo_pts])
    n_pts = np.array([p[1] for p in geo_pts])
    
    # 3. To WGS84 and then to Mosaic Pixels
    from_crs = CRS.from_user_input(ortho_crs)
    transformer = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(e_pts, n_pts)
    
    n_z = 2.0 ** wms_zoom
    wms_pix = []
    for i in range(3):
        x_px = (lons[i] + 180.0) / 360.0 * n_z * 256.0
        lat_rad = math.radians(lats[i])
        y_px = (1.0 - math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) / math.pi) / 2.0 * n_z * 256.0
        wms_pix.append((x_px - wms_bounds_px[0], y_px - wms_bounds_px[1]))
        
    # 4. Affine coefficients
    x0, y0 = wms_pix[0] # TL
    x1, y1 = wms_pix[1] # TR (Step in Col)
    x2, y2 = wms_pix[2] # BL (Step in Row)
    
    a = (x1 - x0) / out_w
    d = (y1 - y0) / out_w
    b = (x2 - x0) / out_h
    e = (y2 - y0) / out_h
    
    matrix = cp.array([[e, d], [b, a]], dtype=cp.float32)
    offset = cp.array([y0, x0], dtype=cp.float32)
    
    # 5. Transform
    result_planar = cp.zeros((4, out_h, out_w), dtype=cp.uint8)
    for i in range(4):
        chan = wms_texture_planar[i] if i < wms_texture_planar.shape[0] else cp.full(wms_texture_planar.shape[1:], 255, dtype=cp.uint8)
        ndimage.affine_transform(
            chan, matrix, offset=offset, output_shape=(out_h, out_w),
            output=result_planar[i], order=1, mode='constant', cval=0, prefilter=False
        )
        
    return cp.transpose(result_planar, (1, 2, 0))


# --- Main Rendering Context ---

class GPURenderContext:
    def __init__(self):
        self.ortho_texture: Optional[cp.ndarray] = None
        self.mipmaps: List[cp.ndarray] = []
        self.ortho_transform: Optional[Affine] = None
        self.ortho_crs = None
        self.ortho_res_m = 1.0 
        self.vector_texture: Optional[cp.ndarray] = None
        self.wms_texture: Optional[cp.ndarray] = None
        self.wms_bounds_px = None 
        self.wms_zoom = None
        self.is_ready = False

    def clear(self):
        self.ortho_texture = None
        self.mipmaps = []
        self.vector_texture = None
        self.wms_texture = None
        self.is_ready = False
        import gc
        gc.collect()
        if HAS_GPU:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except: pass

    def preload(self, dataset, center_points, margin_m, vectors=None, 
                arrow_size=100, cone_len=200, wms_source="google_hybrid", 
                icon_opacity=0.4, progress_callback=None):
        
        def notify(pct, msg):
            if progress_callback: progress_callback(pct, msg)
            print(f"[GPU] {pct}% - {msg}")
            
        if not HAS_GPU: return False
        
        es = [p[0] for p in center_points]
        ns = [p[1] for p in center_points]
        xmin, xmax = min(es) - margin_m, max(es) + margin_m
        ymin, ymax = min(ns) - margin_m, max(ns) + margin_m
        
        window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
        tw, th = int(window.width), int(window.height)
        MAX_DIM = 16384
        if tw > MAX_DIM or th > MAX_DIM:
            scale = MAX_DIM / max(tw, th)
            tw, th = int(tw * scale), int(th * scale)
        
        notify(10, f"Cargando Ortofoto {tw}x{th}...")
        raw_data = dataset.read(
            window=window, out_shape=(dataset.count, th, tw),
            resampling=Resampling.bilinear, boundless=True, fill_value=dataset.nodata
        )
        rgb, alpha = _to_rgba(raw_data, nodata_val=dataset.nodata)
        normalized = _normalize_rgba(rgb, alpha)
        
        self.ortho_transform = rasterio.windows.transform(window, dataset.transform) * Affine.scale(window.width/tw, window.height/th)
        self.ortho_crs = dataset.crs
        self.ortho_res_m = dataset.res[0] * (window.width/tw)
        self.ortho_texture = cp.ascontiguousarray(cp.asarray(normalized).transpose(2, 0, 1))
        
        notify(40, "Generando Mipmaps GPU...")
        self.mipmaps = [self.ortho_texture]
        if tw > 2 and th > 2:
            self.mipmaps.append(cp.ascontiguousarray(self.ortho_texture[:, ::2, ::2]))
        if tw > 4 and th > 4:
            self.mipmaps.append(cp.ascontiguousarray(self.mipmaps[-1][:, ::2, ::2]))
            
        notify(60, "Procesando Vectores...")
        vec_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(vec_img)
        itf = ~self.ortho_transform
        def map_func(e, n): return itf * (e, n)
        
        if vectors:
            for geoms, color, width, pattern in vectors:
                eff_width = max(1, int(width * 1.5))
                for g in geoms:
                    _draw_geometry_precise(draw, g, map_func, color, eff_width, pattern)
        
        self.vector_texture = cp.ascontiguousarray(cp.asarray(np.array(vec_img)).transpose(2, 0, 1))
        
        notify(75, "Descargando WMS...")
        w_geo, s_geo, e_geo, n_geo = transform_bounds(dataset.crs, "EPSG:4326", xmin, ymin, xmax, ymax)
        span_deg = max(e_geo - w_geo, n_geo - s_geo)
        # Use same logic as CPU for zoom
        TARGET_PX = 3840
        wms_zoom = min(19, max(1, int(math.log2((TARGET_PX * 360) / (256 * span_deg)) + 0.5)))
        
        ret_wms = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, wms_zoom, source=wms_source)
        if ret_wms[0]:
            wms_rgba = np.array(ret_wms[0].convert("RGBA"))
            self.wms_texture = cp.ascontiguousarray(cp.asarray(wms_rgba).transpose(2, 0, 1))
            self.wms_bounds_px = ret_wms[1]
            self.wms_zoom = wms_zoom
            
        notify(100, "Precarga GPU completada.")
        self.is_ready = True
        return True

# --- Public API ---

_CONTEXT = GPURenderContext()

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
    wms_source: str = "google_hybrid",
) -> Any:
    
    if not HAS_GPU: raise RuntimeError("GPU not available.")
    if not _CONTEXT.is_ready: return cp.zeros((height, width, 4), dtype=cp.uint8)

    ss_factor = 2
    sw, sh = width * ss_factor, height * ss_factor
    m_per_px_out = (map_half_width_m * 2.0) / width / ss_factor
    
    scale_ratio = m_per_px_out / _CONTEXT.ortho_res_m
    level = 2 if scale_ratio >= 4.0 and len(_CONTEXT.mipmaps) > 2 else (1 if scale_ratio >= 2.0 and len(_CONTEXT.mipmaps) > 1 else 0)
        
    # Render Layers
    if _CONTEXT.wms_texture is not None:
        final_gpu = _sample_wms_layer_gpu_approx(
            _CONTEXT.wms_texture, _CONTEXT.ortho_crs, center_e, center_n, heading, m_per_px_out, sw, sh,
            _CONTEXT.wms_zoom, _CONTEXT.wms_bounds_px
        )
    else:
        final_gpu = cp.zeros((sh, sw, 4), dtype=cp.uint8)
        
    if _CONTEXT.mipmaps:
        ortho_layer = _sample_using_inverse_transform(
            _CONTEXT.mipmaps[level], center_e, center_n, heading, m_per_px_out, sh, sw, _CONTEXT.ortho_transform, level
        )
        final_gpu = _alpha_composite_gpu(ortho_layer, final_gpu)
        
    if _CONTEXT.vector_texture is not None:
        vec_layer = _sample_using_inverse_transform(
            _CONTEXT.vector_texture, center_e, center_n, heading, m_per_px_out, sh, sw, _CONTEXT.ortho_transform, 0
        )
        final_gpu = _alpha_composite_gpu(vec_layer, final_gpu)
        
    # Finalize
    final_gpu = _gpu_downsample_box(final_gpu, ss_factor)
    
    # UI Overlay (CPU functions used for paraty, then uploaded)
    ui_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    center_px = (width // 2, height // 2)
    if cone_opacity > 0:
        _draw_cone(ui_layer, center_px, 0.0, cone_angle_deg, cone_length_px, cone_opacity)
    _draw_center_icon(ui_layer, center_px, arrow_size_px, icon_circle_opacity, icon_circle_size_px, 0.0)
    if show_compass:
        c_sz = max(15, compass_size_px)
        c_pos = (width - (c_sz // 2 + 15), c_sz // 2 + 15)
        _draw_compass(ui_layer, c_pos, c_sz, -heading)
        
    ui_gpu = cp.asarray(np.array(ui_layer))
    return _alpha_composite_gpu(ui_gpu, final_gpu)

def preload_track_gpu(config: Any, jobs: List[Tuple], progress_callback=None) -> None:
    if not HAS_GPU: return
    centers = [(j[1], j[2]) for j in jobs]
    with rasterio.open(config.ortho_path) as dataset:
        vectors = load_vectors(
            dataset.crs, [layer.model_dump() if hasattr(layer, 'model_dump') else layer for layer in config.vector_layers],
            config.vectors_paths, config.curves_path, config.line_color, config.line_width, config.boundary_color,
            config.boundary_width, config.point_color,
        )
        _CONTEXT.clear()
        _CONTEXT.preload(
            dataset=dataset, center_points=centers, margin_m=config.map_half_width_m * 2.5, vectors=vectors,
            arrow_size=config.arrow_size_px, cone_len=config.cone_length_px, wms_source=config.wms_source,
            icon_opacity=getattr(config, 'icon_circle_opacity', 0.4), progress_callback=progress_callback
        )

def cleanup_gpu():
    if _CONTEXT: _CONTEXT.clear()