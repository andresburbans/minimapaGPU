
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

def _sample_using_inverse_transform(
    texture: cp.ndarray,
    center_e: float,
    center_n: float,
    heading: float,
    m_per_px: float,
    out_h: int,
    out_w: int,
    ortho_transform: Affine
) -> cp.ndarray:
    """
    Sample from texture using direct inverse transform approach.
    FIX: Uses -heading to ensure Camera Up aligns with Travel Direction.
    """
    # Rotation Correction: 
    # We want top of screen (negative screen Y) to correspond to "Target Heading".
    # Analysis showed we need to negate heading for standard geometric rotation.
    rad = math.radians(-heading) 
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    
    # Create coordinate grids on GPU
    # Centered grid
    tc, tr = cp.meshgrid(
        cp.arange(out_w, dtype=cp.float64) - out_w / 2.0,
        cp.arange(out_h, dtype=cp.float64) - out_h / 2.0
    )
    
    # Geo offset from center (unrotated)
    geo_dx = tc * m_per_px
    geo_dy = -tr * m_per_px  # Screen Y increases down, Geo N increases up.
    
    # Apply rotation
    geo_e = center_e + geo_dx * cos_h - geo_dy * sin_h
    geo_n = center_n + geo_dx * sin_h + geo_dy * cos_h
    
    # Geo (E, N) -> Input Pixel (c, r) via inverse transform
    inv_tf = ~ortho_transform
    
    src_c = inv_tf.a * geo_e + inv_tf.b * geo_n + inv_tf.c
    src_r = inv_tf.d * geo_e + inv_tf.e * geo_n + inv_tf.f
    
    # Stack coordinates for map_coordinates: (2, H, W) -> (row, col)
    # Ensure float32 output for map_coordinates if needed, or keep float64
    coordinates = cp.stack([src_r, src_c])
    
    # Sample each channel
    texture_c = texture
    if texture.shape[2] == 4:
         texture_c = cp.transpose(texture, (2, 0, 1)) # (C, H, W)
    
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
        
    return cp.transpose(result_c, (1, 2, 0)) # Return (H, W, 4)

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
    BLAZING FAST WMS SAMPLER using Affine Approximation.
    Instead of calculating pyproj per pixel (millions of ops on CPU),
    we calculate it for 4 corners and derive an affine matrix for GPU.
    """
    rad = math.radians(-heading) # Consistency with ortho
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    
    # 1. Define 4 corners of the output view (relative to center pixels)
    # TopLeft, TopRight, BottomRight, BottomLeft
    corners_px = [
        (-out_w/2, -out_h/2),
        (out_w/2, -out_h/2),
        (out_w/2, out_h/2),
        (-out_w/2, out_h/2)
    ]
    
    # 2. Calculate their Geo Coordinates (E, N)
    geo_pts = []
    for px_x, px_y in corners_px:
        geo_dx = px_x * m_per_px
        geo_dy = -px_y * m_per_px
        ge = center_e + geo_dx * cos_h - geo_dy * sin_h
        gn = center_n + geo_dx * sin_h + geo_dy * cos_h
        geo_pts.append((ge, gn))
        
    # 3. Transform to WMS Pixels on CPU (Only 4 points! Fast!)
    e_cpu = np.array([p[0] for p in geo_pts])
    n_cpu = np.array([p[1] for p in geo_pts])
    
    # Initialize transformer (cached implicitly if called often, or cheap enough)
    # Using 'always_xy=True' ensures lon, lat order
    from_crs = CRS.from_user_input(ortho_crs)
    to_crs = CRS("EPSG:4326")
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    
    lons, lats = transformer.transform(e_cpu, n_cpu)
    
    # Convert to WMS Pixels
    n_z = 2.0 ** wms_zoom
    wms_pts = []
    
    for i in range(4):
        lat = lats[i]
        lon = lons[i]
        # Web Mercator Projection
        x_px = (lon + 180.0) / 360.0 * n_z * 256.0
        lat_rad = math.radians(lat)
        val = math.tan(math.pi / 4.0 + lat_rad / 2.0)
        y_px = (1.0 - math.log(val) / math.pi) / 2.0 * n_z * 256.0
        
        # Adjust by WMS texture offset
        wms_ox, wms_oy = wms_bounds_px
        src_x = x_px - wms_ox
        src_y = y_px - wms_oy
        wms_pts.append((src_x, src_y))
        
    # 4. Derive Affine Matrix mapping Output(x, y) -> WMS(u, v)
    # We solve for M such that: [u, v, 1] = M @ [x, y, 1]
    # Since it's an approximation, we can use OpenCV's getAffineTransform or just linear fit
    # We use a simple 3-point solver (TL, TR, BL sufficient for affine)
    # Output Coords (0,0 is Top Left in image space):
    # (0, 0) -> wms_pts[0]
    # (w, 0) -> wms_pts[1]
    # (0, h) -> wms_pts[3]
    
    src_tri = np.array([[0, 0], [out_w, 0], [0, out_h]], dtype=np.float32)
    dst_tri = np.array([wms_pts[0], wms_pts[1], wms_pts[3]], dtype=np.float32)
    
    # Compute Matrix: src -> dst
    # cv2.getAffineTransform equivalent
    # M = dst * src_inv ? No.
    # We want: InputCoord = M @ OutputCoord
    # Wait, ndimage.affine_transform applies Inverse Mapping: Output -> Input
    # So we want matrix M that maps Output(r, c) -> Input(r, c) (WMS y, x)
    # The Matrix M derived above maps Output(x,y) to Input(x,y).
    
    # Let's compute it manually to be safe without cv2
    # x' = a*x + b*y + c
    # y' = d*x + e*y + f
    # Using 0,0 -> x0, y0
    # c = x0, f = y0
    # Using w,0 -> x1, y1
    # a*w + c = x1 => a = (x1 - x0) / w
    # d*w + f = y1 => d = (y1 - y0) / w
    # Using 0,h -> x2, y2
    # b*h + c = x2 => b = (x2 - x0) / h
    # e*h + f = y2 => e = (y2 - y0) / h
    
    x0, y0 = wms_pts[0] # TL
    x1, y1 = wms_pts[1] # TR
    x2, y2 = wms_pts[3] # BL
    
    c_off = x0
    f_off = y0
    a_coeff = (x1 - x0) / out_w
    d_coeff = (y1 - y0) / out_w
    b_coeff = (x2 - x0) / out_h
    e_coeff = (y2 - y0) / out_h
    
    # Matrix for ndimage: [[row_r, row_c], [col_r, col_c]]
    # Output coords are (row, col) i.e. (y, x)
    # Input coords desired (y', x')
    # y' (row) = d*x + e*y + f = d*col + e*row + f
    # x' (col) = a*x + b*y + c = a*col + b*row + c
    
    # So:
    # row_in = e_coeff * row_out + d_coeff * col_out + f_off
    # col_in = b_coeff * row_out + a_coeff * col_out + c_off
    
    matrix = cp.array([[e_coeff, d_coeff], [b_coeff, a_coeff]], dtype=cp.float64)
    offset = cp.array([f_off, c_off], dtype=cp.float64)
    
    # 5. Apply Affine Transform on GPU
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
        self.ortho_texture = None # (H, W, 4) GPU array
        self.ortho_transform = None
        self.ortho_crs = None
        
        self.vector_texture = None 
        
        self.wms_texture = None 
        self.wms_bounds_px = None 
        self.wms_zoom = None
        
        self.icon_texture = None
        
        self.cpu_ortho = None 
        self.cpu_ortho_tf = None
        self.ortho_w = 0
        self.ortho_h = 0
        
        self.is_ready = False

    def clear(self):
        self.ortho_texture = None
        self.vector_texture = None
        self.wms_texture = None
        self.icon_texture = None
        self.cpu_ortho = None
        self.is_ready = False
        if HAS_GPU:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except: pass

    def preload(self, dataset, center_points, margin_m, vectors=None, arrow_size=100, cone_len=200, wms_source="google_hybrid"):
        if not HAS_GPU: return False
        
        es = [p[0] for p in center_points]
        ns = [p[1] for p in center_points]
        xmin, xmax = min(es) - margin_m, max(es) + margin_m
        ymin, ymax = min(ns) - margin_m, max(ns) + margin_m
        
        # Load Ortho 
        window = from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
        target_res = dataset.res[0] # Prefer native
        
        tw = int(window.width)
        th = int(window.height)
        
        # Aggressive VRAM usage but safe caps
        MAX_DIM = 16384
        if tw > MAX_DIM or th > MAX_DIM:
            scale = MAX_DIM / max(tw, th)
            tw = int(tw * scale)
            th = int(th * scale)
        
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
        wms_zoom = min(19, max(12, int(math.log2((4096 * 360) / (256 * span_deg)))))
        
        print(f"[GPU] Fetching WMS Zoom {wms_zoom}...")
        ret_wms = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, wms_zoom, source=wms_source)
        if ret_wms[0]:
            self.wms_texture = cp.asarray(np.array(ret_wms[0].convert("RGBA")))
            self.wms_bounds_px = ret_wms[1]
            self.wms_zoom = wms_zoom
            
        # Icons
        icon_sz = 256
        ic_img = Image.new("RGBA", (icon_sz, icon_sz), (0,0,0,0))
        _draw_center_icon(ic_img, (icon_sz//2, icon_sz//2), arrow_size, 0.4, arrow_size//3, 0)
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
    
    if _CONTEXT.is_ready and _CONTEXT.ortho_texture is not None:
        ortho_layer = _sample_using_inverse_transform(
            texture=_CONTEXT.ortho_texture,
            center_e=center_e,
            center_n=center_n,
            heading=heading,
            m_per_px=m_per_px,
            out_h=sh,
            out_w=sw,
            ortho_transform=_CONTEXT.cpu_ortho_tf
        )
        
        if _CONTEXT.vector_texture is not None:
             vec_layer = _sample_using_inverse_transform(
                texture=_CONTEXT.vector_texture,
                center_e=center_e,
                center_n=center_n,
                heading=heading,
                m_per_px=m_per_px,
                out_h=sh,
                out_w=sw,
                ortho_transform=_CONTEXT.cpu_ortho_tf
            )
             v_alpha = vec_layer[:,:,3:4].astype(cp.float32) / 255.0
             ortho_layer[:,:,:3] = (vec_layer[:,:,:3] * v_alpha + ortho_layer[:,:,:3] * (1.0 - v_alpha)).astype(cp.uint8)
             ortho_layer[:,:,3] = cp.maximum(ortho_layer[:,:,3], vec_layer[:,:,3])
    else:
        ortho_layer = cp.zeros((sh, sw, 4), dtype=cp.uint8)
        
    if _CONTEXT.is_ready and _CONTEXT.wms_texture is not None:
        # Use APX (Approximation) for Speed
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
        
        o_alpha = ortho_layer[:,:,3:4].astype(cp.float32) / 255.0
        final_rgb = (ortho_layer[:,:,:3] * o_alpha + wms_layer[:,:,:3] * (1.0 - o_alpha)).astype(cp.uint8)
        final_gpu = cp.dstack((final_rgb, cp.full((sh, sw, 1), 255, dtype=cp.uint8)))
    else:
        final_gpu = ortho_layer

    final_f = final_gpu.astype(cp.float32)
    final_out = (final_f[0::2, 0::2] + final_f[1::2, 0::2] + final_f[0::2, 1::2] + final_f[1::2, 1::2]) / 4.0
    final_gpu_out = final_out.astype(cp.uint8)

    if _CONTEXT.icon_texture is not None:
        ih, iw, _ = _CONTEXT.icon_texture.shape
        scy, scx = height//2, width//2
        sy, sx = scy - ih//2, scx - iw//2
        if sy >= 0 and sx >= 0 and sy+ih < height and sx+iw < width:
             patch = final_gpu_out[sy:sy+ih, sx:sx+iw].astype(cp.float32)
             ic = _CONTEXT.icon_texture.astype(cp.float32)
             alpha = (ic[:,:,3:4] / 255.0) * icon_circle_opacity
             res = ic[:,:,:3] * alpha + patch[:,:,:3] * (1.0 - alpha)
             final_gpu_out[sy:sy+ih, sx:sx+iw, :3] = res.astype(cp.uint8)

    result = Image.fromarray(cp.asnumpy(final_gpu_out), "RGBA")
    
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
        _CONTEXT.preload(
            dataset=dataset,
            center_points=centers,
            margin_m=margin,
            vectors=vectors,
            arrow_size=config.arrow_size_px,
            cone_len=config.cone_length_px,
            wms_source=config.wms_source
        )

def cleanup_gpu():
    """Interface for app.py"""
    if _CONTEXT:
        _CONTEXT.clear()
