
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
    pass

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
    Proper alpha compositing matching PIL.Image.alpha_composite.
    """
    fg_alpha = fg[:, :, 3:4].astype(cp.float32) / 255.0
    bg_alpha = bg[:, :, 3:4].astype(cp.float32) / 255.0
    
    out_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)
    # Avoid div by zero
    out_alpha_safe = cp.where(out_alpha > 0, out_alpha, 1.0)
    
    fg_rgb = fg[:, :, :3].astype(cp.float32)
    bg_rgb = bg[:, :, :3].astype(cp.float32)
    
    out_rgb = (fg_rgb * fg_alpha + 
               bg_rgb * bg_alpha * (1.0 - fg_alpha)) / out_alpha_safe
    
    out_rgb = cp.clip(out_rgb, 0, 255).astype(cp.uint8)
    out_alpha_u8 = (out_alpha * 255).astype(cp.uint8)
    
    return cp.dstack((out_rgb, out_alpha_u8))

def _gpu_downsample_lanczos(arr: cp.ndarray, scale: int) -> cp.ndarray:
    """
    High-quality downsampling (Box filter for integer scale).
    """
    if scale == 1:
        return arr
    
    if scale == 2:
         # Box filter 2x2
         arr_f = arr.astype(cp.float32)
         out = (arr_f[0::2, 0::2] + arr_f[1::2, 0::2] + arr_f[0::2, 1::2] + arr_f[1::2, 1::2]) / 4.0
         return out.astype(cp.uint8)
         
    # Fallback to zoom
    factor = 1.0 / scale
    return zoom(arr, (factor, factor, 1), order=1).astype(cp.uint8)

def _sample_using_inverse_transform(
    texture: cp.ndarray,
    center_e: float,
    center_n: float,
    heading: float,
    m_per_px: float,
    out_h: int,
    out_w: int,
    ortho_transform: Affine,
    mipmap_level: int = 0
) -> cp.ndarray:
    """
    Sample texture using direct inverse transform with Mipmap support.
    """
    rad = math.radians(-heading) 
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    
    # Coordinate grids (Float64 for precision)
    tc, tr = cp.meshgrid(
        cp.arange(out_w, dtype=cp.float64) - out_w / 2.0,
        cp.arange(out_h, dtype=cp.float64) - out_h / 2.0
    )
    
    geo_dx = tc * m_per_px
    geo_dy = -tr * m_per_px 
    
    geo_e = center_e + geo_dx * cos_h - geo_dy * sin_h
    geo_n = center_n + geo_dx * sin_h + geo_dy * cos_h
    
    inv_tf = ~ortho_transform
    
    src_c = inv_tf.a * geo_e + inv_tf.b * geo_n + inv_tf.c
    src_r = inv_tf.d * geo_e + inv_tf.e * geo_n + inv_tf.f
    
    # Apply Mipmap Scaling
    if mipmap_level > 0:
        scale = 1.0 / (2 ** mipmap_level)
        src_c *= scale
        src_r *= scale
    
    coordinates = cp.stack([src_r, src_c])
    
    texture_c = texture
    if texture.shape[2] == 4:
         texture_c = cp.transpose(texture, (2, 0, 1))
    
    result_c = cp.zeros((4, out_h, out_w), dtype=cp.uint8)
    
    for i in range(4):
        ndimage.map_coordinates(
            texture_c[i],
            coordinates,
            output=result_c[i],
            order=1, # Bilinear
            mode='constant',
            cval=0,
            prefilter=False
        )
        
    return cp.transpose(result_c, (1, 2, 0))

def _sample_wms_layer_gpu_approx(
    wms_texture: cp.ndarray,
    ortho_crs,
    center_e: float,
    center_n: float,
    heading: float,
    m_per_px: float,
    out_h: int,
    out_w: int,
    wms_zoom: int,
    wms_bounds_px: Tuple[float, float]
) -> cp.ndarray:
    """
    Fast WMS sampling using Affine Approx.
    """
    rad = math.radians(-heading)
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    
    corners_px = [
        (-out_w/2, -out_h/2),
        (out_w/2, -out_h/2),
        (out_w/2, out_h/2),
        (-out_w/2, out_h/2)
    ]
    
    geo_pts = []
    for px_x, px_y in corners_px:
        geo_dx = px_x * m_per_px
        geo_dy = -px_y * m_per_px
        ge = center_e + geo_dx * cos_h - geo_dy * sin_h
        gn = center_n + geo_dx * sin_h + geo_dy * cos_h
        geo_pts.append((ge, gn))
        
    e_cpu = np.array([p[0] for p in geo_pts])
    n_cpu = np.array([p[1] for p in geo_pts])
    
    from_crs = CRS.from_user_input(ortho_crs)
    to_crs = CRS("EPSG:4326")
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    
    lons, lats = transformer.transform(e_cpu, n_cpu)
    
    n_z = 2.0 ** wms_zoom
    wms_pts = []
    
    for i in range(4):
        lat = lats[i]
        lon = lons[i]
        x_px = (lon + 180.0) / 360.0 * n_z * 256.0
        lat_rad = math.radians(lat)
        val = math.tan(math.pi / 4.0 + lat_rad / 2.0)
        y_px = (1.0 - math.log(val) / math.pi) / 2.0 * n_z * 256.0
        wms_ox, wms_oy = wms_bounds_px
        src_x = x_px - wms_ox
        src_y = y_px - wms_oy
        wms_pts.append((src_x, src_y))
        
    x0, y0 = wms_pts[0]
    x1, y1 = wms_pts[1]
    x2, y2 = wms_pts[3]
    
    c_off = x0
    f_off = y0
    a_coeff = (x1 - x0) / out_w
    d_coeff = (y1 - y0) / out_w
    b_coeff = (x2 - x0) / out_h
    e_coeff = (y2 - y0) / out_h
    
    matrix = cp.array([[e_coeff, d_coeff], [b_coeff, a_coeff]], dtype=cp.float64)
    offset = cp.array([f_off, c_off], dtype=cp.float64)
    
    wms_c = wms_texture
    if wms_texture.shape[2] == 4:
         wms_c = cp.transpose(wms_texture, (2, 0, 1))
         
    result_c = cp.zeros((4, out_h, out_w), dtype=cp.uint8)
    
    for i in range(4):
        if i < wms_c.shape[0]:
             chan = wms_c[i]
        else:
            chan = cp.full((wms_c.shape[1], wms_c.shape[2]), 255, dtype=cp.uint8)
            
        ndimage.affine_transform(
            chan,
            matrix,
            offset=offset,
            output_shape=(out_h, out_w),
            output=result_c[i],
            order=1,
            mode='constant',
            cval=0,
            prefilter=False
        )
        
    return cp.transpose(result_c, (1, 2, 0))


class GPURenderContext:
    def __init__(self):
        self.ortho_texture = None
        self.mipmaps = [] # List of textures: [L0, L1, L2]
        self.ortho_transform = None
        self.ortho_crs = None
        self.ortho_res_m = 1.0 
        
        self.vector_texture = None 
        self.wms_texture = None 
        self.wms_bounds_px = None 
        self.wms_zoom = None
        self.icon_texture = None
        self.cpu_ortho_tf = None
        self.ortho_w = 0
        self.ortho_h = 0
        self.is_ready = False

    def clear(self):
        self.ortho_texture = None
        self.mipmaps = []
        self.vector_texture = None
        self.wms_texture = None
        self.icon_texture = None
        self.is_ready = False
        
        # Force Python Garbage Collection to release references
        import gc
        gc.collect()
        
        if HAS_GPU:
            try:
                # Free standard memory pool
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                # Free pinned memory pool (used for CPU<->GPU transfers)
                pinned_pool = cp.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()
            except: pass


    def preload(self, dataset, center_points, margin_m, vectors=None, arrow_size=100, cone_len=200, wms_source="google_hybrid", icon_opacity=0.4):
        if not HAS_GPU: return False
        
        es = [p[0] for p in center_points]
        ns = [p[1] for p in center_points]
        xmin, xmax = min(es) - margin_m, max(es) + margin_m
        ymin, ymax = min(ns) - margin_m, max(ns) + margin_m
        
        window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
        target_res = dataset.res[0] 
        self.ortho_res_m = target_res
        
        tw = int(window.width)
        th = int(window.height)
        
        # Max Size 16k
        MAX_DIM = 16384
        if tw > MAX_DIM or th > MAX_DIM:
            scale = MAX_DIM / max(tw, th)
            tw = int(tw * scale)
            th = int(th * scale)
            self.ortho_res_m /= scale 
        
        print(f"[GPU] Loading Ortho {tw}x{th}...")
        
        cpu_data = dataset.read(
            window=window,
            out_shape=(dataset.count, th, tw),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=dataset.nodata
        )
        
        rgb, alpha = _to_rgba(cpu_data, nodata_val=dataset.nodata)
        normalized = _normalize_rgba(rgb, alpha)
        
        self.cpu_ortho_tf = rasterio.windows.transform(window, dataset.transform) * Affine.scale(window.width/tw, window.height/th)
        self.ortho_crs = dataset.crs
        self.ortho_w = tw
        self.ortho_h = th
        
        self.ortho_texture = cp.asarray(normalized)
        
        # Generate Mipmaps
        print("[GPU] Generating Mipmaps...")
        self.mipmaps = [self.ortho_texture]
        # Level 1 (Half)
        if tw > 2 and th > 2:
            l1 = self.ortho_texture[::2, ::2, :]
            self.mipmaps.append(l1)
        # Level 2 (Quarter)
        if tw > 4 and th > 4:
            l2 = self.mipmaps[-1][::2, ::2, :]
            self.mipmaps.append(l2)
        
        # Bake Vectors
        print(f"[GPU] Baking Vectors...")
        vec_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(vec_img)
        inv_tf = ~self.cpu_ortho_tf
        def vec_tf(e, n):
            c, r = inv_tf * (e, n)
            return c, r
        if vectors:
            for geom_iter, color, width, pattern in vectors:
                eff_width = max(1, int(width * 1.5)) 
                for geom in geom_iter:
                    _draw_geometry_precise(draw, geom, vec_tf, color, eff_width, pattern)
        
        self.vector_texture = cp.asarray(np.array(vec_img))
        
        # WMS
        w_geo, s_geo, e_geo, n_geo = transform_bounds(dataset.crs, "EPSG:4326", xmin, ymin, xmax, ymax)
        span_deg = max(e_geo - w_geo, n_geo - s_geo)
        
        # Limit WMS Texture Size to avoid hitting tile limits (225 tiles approx 3840x3840px)
        # We target approx 4000px max dimension to be safe
        TARGET_WMS_DIM = 3800
        wms_zoom = min(19, max(1, int(math.log2((TARGET_WMS_DIM * 360) / (256 * span_deg)))))
        
        print(f"[GPU] Fetching WMS Zoom {wms_zoom} for span {span_deg:.4f}...")
        ret_wms = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, wms_zoom, source=wms_source)
        
        # Retry with lower zoom if failed (likely due to tile limit or network)
        if not ret_wms[0] and wms_zoom > 10:
             print(f"[GPU] WMS Fetch failed, retrying with Zoom {wms_zoom-2}...")
             wms_zoom -= 2
             ret_wms = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, wms_zoom, source=wms_source)

        if ret_wms[0]:
            print(f"[GPU] WMS Loaded: {ret_wms[0].size}")
            self.wms_texture = cp.asarray(np.array(ret_wms[0].convert("RGBA")))
            self.wms_bounds_px = ret_wms[1]
            self.wms_zoom = wms_zoom
        else:
            print("[GPU] Warning: WMS Layer could not be loaded. Background will be black.")
            self.wms_texture = None
            
        # Icons
        icon_sz = 256
        ic_img = Image.new("RGBA", (icon_sz, icon_sz), (0,0,0,0))
        # Use provided icon_opacity
        _draw_center_icon(ic_img, (icon_sz//2, icon_sz//2), arrow_size, icon_opacity, arrow_size//3, 0)
        self.icon_texture = cp.asarray(np.array(ic_img))
        
        self.is_ready = True
        return True

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
) -> Image.Image:
    
    if not HAS_GPU:
        raise RuntimeError("GPU acceleration requested but CuPy is not available.")

    ss_factor = 2
    sw, sh = width * ss_factor, height * ss_factor
    m_per_px = (map_half_width_m * 2.0) / width / ss_factor
    
    final_gpu = None
    
    # 1. Sample Ortho w/ Mipmaps
    if _CONTEXT.is_ready and _CONTEXT.mipmaps:
        if _CONTEXT.ortho_res_m > 0:
            scale_ratio = m_per_px / _CONTEXT.ortho_res_m
            if scale_ratio >= 4.0 and len(_CONTEXT.mipmaps) > 2:
                level = 2
            elif scale_ratio >= 2.0 and len(_CONTEXT.mipmaps) > 1:
                level = 1
            else:
                level = 0
        else:
            level = 0
            
        ortho_layer = _sample_using_inverse_transform(
            texture=_CONTEXT.mipmaps[level],
            center_e=center_e,
            center_n=center_n,
            heading=heading,
            m_per_px=m_per_px,
            out_h=sh,
            out_w=sw,
            ortho_transform=_CONTEXT.cpu_ortho_tf,
            mipmap_level=level
        )
        
        # 2. Composite Vectors
        if _CONTEXT.vector_texture is not None:
             vec_layer = _sample_using_inverse_transform(
                texture=_CONTEXT.vector_texture,
                center_e=center_e,
                center_n=center_n,
                heading=heading,
                m_per_px=m_per_px,
                out_h=sh,
                out_w=sw,
                ortho_transform=_CONTEXT.cpu_ortho_tf,
                mipmap_level=0 # Always Level 0 for Sharp Vectors
            )
             ortho_layer = _alpha_composite_gpu(vec_layer, ortho_layer)
    else:
        ortho_layer = cp.zeros((sh, sw, 4), dtype=cp.uint8)
        
    # 3. Composite WMS
    if _CONTEXT.is_ready and _CONTEXT.wms_texture is not None:
        wms_layer = _sample_wms_layer_gpu_approx(
            wms_texture=_CONTEXT.wms_texture,
            ortho_crs=_CONTEXT.ortho_crs,
            center_e=center_e,
            center_n=center_n,
            heading=heading,
            m_per_px=m_per_px,
            out_h=sh,
            out_w=sw,
            wms_zoom=_CONTEXT.wms_zoom,
            wms_bounds_px=_CONTEXT.wms_bounds_px
        )
        ortho_layer = _alpha_composite_gpu(ortho_layer, wms_layer)
    
    # 4. Downsample
    final_gpu = _gpu_downsample_lanczos(ortho_layer, ss_factor)

    # 5. Icons
    if _CONTEXT.icon_texture is not None:
        ih, iw, _ = _CONTEXT.icon_texture.shape
        scy, scx = height//2, width//2
        sy, sx = scy - ih//2, scx - iw//2
        if sy >= 0 and sx >= 0 and sy+ih < height and sx+iw < width:
             patch = final_gpu[sy:sy+ih, sx:sx+iw]
             ic = _CONTEXT.icon_texture
             # Icon composite NO MULTIPLICATION
             res = _alpha_composite_gpu(ic, patch)
             final_gpu[sy:sy+ih, sx:sx+iw] = res

    result = Image.fromarray(cp.asnumpy(final_gpu), "RGBA")
    
    if show_compass:
        c_pos = (width - compass_size_px - 10, compass_size_px + 10)
        _draw_compass(result, c_pos, compass_size_px, -heading)
        
    return result

def preload_track_gpu(config: Any, jobs: List[Tuple]) -> None:
    if not HAS_GPU: return
    
    centers = [(j[1], j[2]) for j in jobs]
    
    with rasterio.open(config.ortho_path) as dataset:
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
        
        margin = config.map_half_width_m * 2.5
        _CONTEXT.clear()
        
        # Safely get opacity
        icon_op = getattr(config, 'icon_circle_opacity', 0.4)
        
        _CONTEXT.preload(
            dataset=dataset,
            center_points=centers,
            margin_m=margin,
            vectors=vectors,
            arrow_size=config.arrow_size_px,
            cone_len=config.cone_length_px,
            wms_source=config.wms_source,
            icon_opacity=icon_op
        )

def cleanup_gpu():
    if _CONTEXT:
        _CONTEXT.clear()
