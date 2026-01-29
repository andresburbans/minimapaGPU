import io
import math
import urllib.request
import functools
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Any
from multiprocessing import get_context
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from models import PreviewRequest, RenderConfig, TrackRequest, OverlayRequest
from render import (
    load_segments,
    interpolate_position,
    load_vectors,
    clip_vectors,
    render_frame,
    init_worker,
    render_frame_job,
)
import gpu_utils  # Auto-detecta y configura GPU al importar
try:
    from render_gpu import (
        render_frame_gpu,
        init_gpu,
        preload_track_gpu,
        cleanup_gpu,
    )
    GPU_RENDER_AVAILABLE = True
except ImportError:
    GPU_RENDER_AVAILABLE = False
    def render_frame_gpu(*args, **kwargs):
        raise NotImplementedError("GPU render not available")


from track import track_points, render_overlay

import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TEMP_DIR = ROOT / "temp"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOB_LOCK = threading.Lock()
_JOBS = {}


def cleanup_temp_files(max_age_hours: float = 0) -> dict:
    """Clean up temporary files older than max_age_hours.
    
    Args:
        max_age_hours: Delete files older than this. Use 0 to delete all temp files.
    
    Returns:
        Dictionary with cleanup statistics.
    """
    import time
    stats = {"deleted_files": 0, "deleted_dirs": 0, "freed_bytes": 0, "errors": []}
    
    if not TEMP_DIR.exists():
        return stats
    
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    # Delete frame directories
    for item in TEMP_DIR.iterdir():
        try:
            if item.is_dir() and item.name.startswith("frames_"):
                should_delete = max_age_hours == 0
                if not should_delete:
                    mtime = item.stat().st_mtime
                    should_delete = (now - mtime) > max_age_seconds
                
                if should_delete:
                    # Count files and size before deletion
                    for f in item.rglob("*"):
                        if f.is_file():
                            stats["freed_bytes"] += f.stat().st_size
                            stats["deleted_files"] += 1
                    shutil.rmtree(item, ignore_errors=True)
                    stats["deleted_dirs"] += 1
        except Exception as e:
            stats["errors"].append(str(e))
    
    return stats


# Clean up temporary files on startup
@app.on_event("startup")
async def startup_cleanup():
    """Clean all temporary files when the server starts."""
    stats = cleanup_temp_files(max_age_hours=0)  # Delete all temp files
    freed_mb = stats["freed_bytes"] / (1024 * 1024)
    if stats["deleted_files"] > 0:
        print(f"[CLEANUP] Startup cleanup: deleted {stats['deleted_files']} files ({freed_mb:.1f} MB) in {stats['deleted_dirs']} directories")


class UploadResponse(BaseModel):
    path: str
    filename: str


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    DATA_DIR.mkdir(exist_ok=True)
    suffix = Path(file.filename).suffix
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = DATA_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return UploadResponse(path=str(dest), filename=file.filename)


@app.post("/cleanup")
async def cleanup_endpoint(max_age_hours: float = 0):
    """Clean up temporary frame files.
    
    Args:
        max_age_hours: Delete files older than this (0 = all files)
    """
    stats = cleanup_temp_files(max_age_hours)
    freed_mb = stats["freed_bytes"] / (1024 * 1024)
    return {
        "success": True,
        "message": f"Deleted {stats['deleted_files']} files ({freed_mb:.1f} MB) in {stats['deleted_dirs']} directories",
        "stats": stats,
    }


@app.post("/cleanup/data")
async def cleanup_data_endpoint():
    """Clean up uploaded data files (orthos, vectors, etc.)."""
    stats = {"deleted_files": 0, "freed_bytes": 0, "errors": []}
    
    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            try:
                if item.is_file():
                    stats["freed_bytes"] += item.stat().st_size
                    stats["deleted_files"] += 1
                    item.unlink()
                elif item.is_dir():
                    for f in item.rglob("*"):
                        if f.is_file():
                            stats["freed_bytes"] += f.stat().st_size
                            stats["deleted_files"] += 1
                    shutil.rmtree(item, ignore_errors=True)
            except Exception as e:
                stats["errors"].append(str(e))
    
    freed_mb = stats["freed_bytes"] / (1024 * 1024)
    return {
        "success": True,
        "message": f"Deleted {stats['deleted_files']} uploaded files ({freed_mb:.1f} MB)",
        "stats": stats,
    }


def _dispatch_render(cfg: RenderConfig, dataset: rasterio.io.DatasetReader, vectors: list, center_e: float, center_n: float, heading: float, job_id: Optional[str] = None, frame_idx: Optional[int] = None) -> Image.Image:
    """Helper to dispatch rendering to GPU or CPU with automatic fallback."""
    render_params = {
        "dataset": dataset,
        "vectors": vectors,
        "center_e": center_e,
        "center_n": center_n,
        "heading": heading,
        "width": cfg.width,
        "height": cfg.height,
        "map_half_width_m": cfg.map_half_width_m,
        "arrow_size_px": cfg.arrow_size_px,
        "cone_angle_deg": cfg.cone_angle_deg,
        "cone_length_px": cfg.cone_length_px,
        "cone_opacity": cfg.cone_opacity,
        "icon_circle_opacity": cfg.icon_circle_opacity,
        "icon_circle_size_px": cfg.icon_circle_size_px,
        "show_compass": cfg.show_compass,
        "compass_size_px": cfg.compass_size_px,
        "wms_source": cfg.wms_source,
    }

    if cfg.use_gpu and GPU_RENDER_AVAILABLE:
        try:
            return render_frame_gpu(**render_params)
        except Exception as e:
            err_msg = f"[GPU] Error rendering {'preview' if frame_idx is None else f'frame {frame_idx}'}: {e}. Falling back to CPU."
            print(err_msg)
            if job_id:
                _update_job(job_id, log=err_msg)
    
    # Default/Fallback to CPU
    return render_frame(**render_params)


@app.post("/preview")
async def preview(req: PreviewRequest):
    cfg = req.config
    segments = load_segments(cfg.csv_path)
    center_e, center_n, heading = interpolate_position(segments, req.time_sec)

    with rasterio.open(cfg.ortho_path) as dataset:
        vectors = load_vectors(
            dataset.crs,
            [layer.model_dump() for layer in cfg.vector_layers],
            cfg.vectors_paths,
            cfg.curves_path,
            cfg.line_color,
            cfg.line_width,
            cfg.boundary_color,
            cfg.boundary_width,
            cfg.point_color,
        )
        # Use a safe overscan for clipping vectors to avoid truncation during rotation
        # A factor of 2.0 covers the diagonal and rotation safely
        clip_margin = cfg.map_half_width_m * 2.0
        bbox = (
            center_e - clip_margin,
            center_n - clip_margin,
            center_e + clip_margin,
            center_n + clip_margin,
        )
        vectors = clip_vectors(vectors, bbox)
        # Dispatch rendering (handles GPU/CPU switch and fallback)
        frame = _dispatch_render(cfg, dataset, vectors, center_e, center_n, heading)
            
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/render")
async def render(config: RenderConfig):
    segments = load_segments(config.csv_path)
    total_frames = int(config.duration_sec * config.fps)
    output_path = OUTPUT_DIR / config.output_name
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    frame_dir = TEMP_DIR / f"frames_{uuid.uuid4().hex}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx in range(total_frames):
        t = idx / config.fps
        center_e, center_n, heading = interpolate_position(segments, t)
        frame_path = str(frame_dir / f"frame_{idx:06d}.png")
        jobs.append((idx, center_e, center_n, heading, frame_path))

    job_id = uuid.uuid4().hex
    _set_job(job_id, total_frames, str(output_path), "minimap")

    thread = threading.Thread(
        target=_render_task,
        args=(job_id, config, jobs, frame_dir, output_path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/track")
async def track(req: TrackRequest):
    job_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"track_{job_id}.json"
    _set_job(job_id, 0, str(output_path), "tracking")
    thread = threading.Thread(
        target=_track_task,
        args=(job_id, req, output_path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/render-overlay")
async def render_overlay_video(req: OverlayRequest):
    job_id = uuid.uuid4().hex
    output_dir = Path(req.output_dir) if req.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / req.output_name
    _set_job(job_id, 0, str(output_path), "overlay")
    thread = threading.Thread(
        target=_render_overlay_task,
        args=(job_id, req, output_path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="video/mp4", filename=Path(path).name)


@app.get("/file")
async def file(path: str):
    return FileResponse(path)


@app.get("/gpu-info")
async def gpu_info():
    """Endpoint para obtener información de la GPU disponible"""
    return gpu_utils.detect_cuda_gpu()


@app.get("/stats")
async def stats():
    """Endpoint de telemetría en tiempo real"""
    return gpu_utils.get_system_stats()


@app.get("/ortho-info")
async def ortho_info(path: str):
    """Obtener metadata del ortomosaico para el picker interactivo"""
    try:
        with rasterio.open(path) as dataset:
            # Si no tiene overviews, intentamos crearlos para optimizar
            if not dataset.overviews(1):
                try:
                    # Necesitamos abrir en modo r+ para crear overviews
                    with rasterio.open(path, "r+") as ds_edit:
                        factors = [2, 4, 8, 16, 32, 64]
                        ds_edit.build_overviews(factors, Resampling.average)
                        ds_edit.update_tags(ns='rio_utils', overviews='created')
                except Exception as e:
                    print(f"No se pudieron crear overviews: {e}")

            bounds = dataset.bounds
            
            # EXPANSION TRICK:
            # Expandimos 2.0x los bounds reportados para que el frontend cree un canvas más grande
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            half_w = (bounds.right - bounds.left) / 2
            half_h = (bounds.top - bounds.bottom) / 2
            
            exp_factor = 2.0
            new_half_w = half_w * exp_factor
            new_half_h = half_h * exp_factor
            
            new_bounds = [
                center_x - new_half_w,
                center_y - new_half_h,
                center_x + new_half_w,
                center_y + new_half_h
            ]
            
            return {
                "width": int(dataset.width * exp_factor),
                "height": int(dataset.height * exp_factor),
                "bounds": new_bounds,
                "transform": list(dataset.transform)[:6],
                "crs": str(dataset.crs),
                "res": dataset.res
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/ortho-preview")
async def ortho_preview(path: str, max_size: int = 1500):
    """Generar una imagen de preview del ortomosaico para navegación"""
    from PIL import Image
    import numpy as np
    
    try:
        with rasterio.open(path) as dataset:
            # Calcular factor de reducción ideal
            scale = min(max_size / dataset.width, max_size / dataset.height, 1.0)
            out_width = max(int(dataset.width * scale), 1)
            out_height = max(int(dataset.height * scale), 1)
            
            # Leer usando overviews si están disponibles (automático en rasterio.read con out_shape)
            data = dataset.read(
                out_shape=(dataset.count, out_height, out_width),
                resampling=Resampling.bilinear,
            )
            
            # Gestionar bandas
            if data.shape[0] == 4:
                # RGBA
                img_data = np.transpose(data, (1, 2, 0))
                mode = "RGBA"
            elif data.shape[0] >= 3:
                # RGB
                rgb = data[:3]
                transposed = np.transpose(rgb, (1, 2, 0))
                
                # Check for mask if implicit with nodata
                if dataset.nodata is not None:
                     # Create alpha from nodata
                     # Assuming band 1 check is enough or check all
                     mask = (data[0] != dataset.nodata).astype(np.uint8) * 255
                     img_data = np.dstack((transposed, mask))
                     mode = "RGBA"
                else:
                    img_data = transposed
                    mode = "RGB"
            else:
                # Grayscale
                single = data[0]
                rgb = np.stack([single, single, single], axis=-1)
                img_data = rgb
                mode = "RGB"
                if dataset.nodata is not None:
                     mask = (single != dataset.nodata).astype(np.uint8) * 255
                     img_data = np.dstack((rgb, mask))
                     mode = "RGBA"
            
            # Normalización robusta solo en canales de color
            color_channels = 3 if mode == "RGBA" or mode == "RGB" else 1 # Simplifying
            
            # If not uint8, normalize
            if img_data.dtype != np.uint8:
                # Splitting alpha if present
                colors = img_data[..., :3]
                valid_mask = ~np.isnan(colors)
                if np.any(valid_mask):
                    v_min = np.percentile(colors[valid_mask], 2)
                    v_max = np.percentile(colors[valid_mask], 98)
                    if v_max - v_min > 1e-6:
                         colors = ((colors - v_min) / (v_max - v_min) * 255).clip(0, 255)
                    else:
                         colors = np.zeros_like(colors)
                
                colors = colors.astype(np.uint8)
                if mode == "RGBA":
                    alpha = img_data[..., 3].astype(np.uint8)
                    img_data = np.dstack((colors, alpha))
                else:
                    img_data = colors
            
            # 4. Crear lienzo transparente de tamaño expandido
            EXP_FACTOR = 2.0
            
            # Dimensiones deseadas finales (thumbnail)
            # El ortho original escalado a 'out_width/out_height' cabe en max_size
            # Pero queremos devolver una imagen de tamaño max_size que contenga padding.
            
            # Recalculamos: max_size será el tamaño del bounding box expandido.
            # El ortho real será size / EXP_FACTOR
            
            real_w_thumb = int(out_width)
            real_h_thumb = int(out_height)
            
            final_w = int(real_w_thumb * EXP_FACTOR)
            final_h = int(real_h_thumb * EXP_FACTOR)
            
            # Asegurar que no sea gigantesca para la red, limitamos el output final
            if final_w > max_size or final_h > max_size:
                ratio = min(max_size / final_w, max_size / final_h)
                final_w = int(final_w * ratio)
                final_h = int(final_h * ratio)
                real_w_thumb = int(real_w_thumb * ratio)
                real_h_thumb = int(real_h_thumb * ratio)
                
            # Resize ortho to its relative thumb size
            ortho_img = Image.fromarray(img_data, mode=mode)
            ortho_img = ortho_img.resize((max(1, real_w_thumb), max(1, real_h_thumb)), Image.NEAREST)
            
            # Create transparent canvas
            canvas = Image.new("RGBA", (final_w, final_h), (0, 0, 0, 0))
            
            # Paste in center
            offset_x = (final_w - real_w_thumb) // 2
            offset_y = (final_h - real_h_thumb) // 2
            canvas.paste(ortho_img, (offset_x, offset_y)) 
            
            buf = io.BytesIO()
            canvas.save(buf, format="PNG", optimize=True) # PNG supports transparency
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response(content=str(e), status_code=500)


def _clamp_latlon(lat: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, lat))


def _clamp_latlon_bounds(west: float, south: float, east: float, north: float):
    min_lat = -85.05112878
    max_lat = 85.05112878
    south = _clamp_latlon(south, min_lat, max_lat)
    north = _clamp_latlon(north, min_lat, max_lat)
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    return west, south, east, north


def _latlon_to_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    lat = _clamp_latlon(lat, -85.05112878, 85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * 256.0
    return x, y


def _choose_zoom(west: float, south: float, east: float, north: float, max_size: int) -> int:
    for z in range(20, 5, -1):
        px_west, px_north = _latlon_to_pixel(north, west, z)
        px_east, px_south = _latlon_to_pixel(south, east, z)
        span_x = abs(px_east - px_west)
        span_y = abs(px_south - px_north)
        tiles_x = math.ceil(span_x / 256)
        tiles_y = math.ceil(span_y / 256)
        if tiles_x * tiles_y <= 16 and span_x <= max_size * 1.2 and span_y <= max_size * 1.2:
            return z
    return 6


@functools.lru_cache(maxsize=1000)
def _fetch_tile(x: int, y: int, z: int, source: str = "google_hybrid"):
    from PIL import Image
    
    # Same logic as render.py to keep things consistent
    def xyz_to_quadkey(x, y, z):
        quadkey = ""
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quadkey += str(digit)
        return quadkey

    sources = {
        "google_hybrid": f"https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "google_satellite": f"http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        "esri_satellite": f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "bing_satellite": f"http://ecn.t3.tiles.virtualearth.net/tiles/a{xyz_to_quadkey(x, y, z)}.jpeg?g=0"
    }
    
    url = sources.get(source, sources["google_hybrid"])
    req = urllib.request.Request(url, headers={"User-Agent": "MinimapaGPT/1.0"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _fetch_wms_mosaic(west: float, south: float, east: float, north: float, zoom: int, source: str = "google_hybrid"):
    from PIL import Image

    px_west, px_north = _latlon_to_pixel(north, west, zoom)
    px_east, px_south = _latlon_to_pixel(south, east, zoom)
    min_tx = int(px_west // 256)
    max_tx = int(px_east // 256)
    min_ty = int(px_north // 256)
    max_ty = int(px_south // 256)

    tiles_x = max_tx - min_tx + 1
    tiles_y = max_ty - min_ty + 1
    tiles_x = max(1, tiles_x)
    tiles_y = max(1, tiles_y)
    
    if tiles_x * tiles_y > 225:
        return Image.new("RGB", (1, 1), (0,0,0))

    mosaic = Image.new("RGB", (tiles_x * 256, tiles_y * 256), (20, 20, 20))
    for ty in range(min_ty, max_ty + 1):
        for tx in range(min_tx, max_tx + 1):
            try:
                tile = _fetch_tile(tx, ty, zoom, source=source)
            except Exception:
                tile = Image.new("RGB", (256, 256), (30, 30, 30))
            mosaic.paste(tile, ((tx - min_tx) * 256, (ty - min_ty) * 256))

    left = int(px_west - min_tx * 256)
    top = int(px_north - min_ty * 256)
    right = int(px_east - min_tx * 256)
    bottom = int(px_south - min_ty * 256)
    left = max(0, min(mosaic.width, left))
    top = max(0, min(mosaic.height, top))
    right = max(left + 1, min(mosaic.width, right))
    bottom = max(top + 1, min(mosaic.height, bottom))
    return mosaic.crop((left, top, right, bottom))


@app.get("/ortho-wms-preview")
async def ortho_wms_preview(path: Optional[str] = None, max_size: int = 1500, zoom: Optional[int] = None, bounds: Optional[str] = None, source: str = "google_hybrid"):
    """Genera un preview satelital (tiles). Si no hay path, usa bounds (w,s,e,n)."""
    from rasterio.warp import transform_bounds
    from PIL import Image

    try:
        west, south, east, north = 0.0, 0.0, 0.0, 0.0
        
        if path:
            with rasterio.open(path) as dataset:
                b = dataset.bounds
                try:
                    west, south, east, north = transform_bounds(
                        dataset.crs,
                        "EPSG:4326",
                        b.left,
                        b.bottom,
                        b.right,
                        b.top,
                        densify_pts=21,
                    )
                    # EXPANSION WMS:
                    # Expandir el bounding box geográfico al mismo factor que ortho_info
                    cx = (west + east) / 2
                    cy = (south + north) / 2
                    hw = (east - west) / 2
                    hh = (north - south) / 2
                    exp = 2.0
                    west = cx - hw * exp
                    east = cx + hw * exp
                    south = cy - hh * exp
                    north = cy + hh * exp
                except Exception:
                    west, south, east, north = b.left, b.bottom, b.right, b.top
        elif bounds:
            # Formato esperado: "west,south,east,north"
            parts = [float(p) for p in bounds.split(",")]
            if len(parts) == 4:
                west, south, east, north = parts
            else:
                raise ValueError("Bounds invalidos")
        else:
            raise ValueError("Se requiere path o bounds")

        west, south, east, north = _clamp_latlon_bounds(west, south, east, north)
        target_zoom = zoom if zoom is not None else _choose_zoom(west, south, east, north, max_size)
        img = _fetch_wms_mosaic(west, south, east, north, target_zoom, source=source)
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response(content=str(e), status_code=500)


@app.get("/default-metadata")
async def default_metadata():
    """Retorna metadatos predeterminados (Colombia) para iniciar sin archivo."""
    # Colombia aprox bounds in EPSG:3857 (Web Mercator)
    # W: -79, S: -4, E: -67, N: 13
    # Converted roughly to 3857
    return {
        "width": 1920, # Virtual canvas size
        "height": 1080, # 16:9 aspect
        "bounds": [-8794239, -445640, -7458406, 1459456], # EPSG:3857 meters
        "transform": [1000, 0, -8794239, 0, -1000, 1459456], # Dummy transform
        "crs": "EPSG:3857",
        "res": [1000, 1000],
        # También enviamos los bounds lat/lon para llamar al WMS
        "wms_bounds": "-79,-4,-67,13"
    }


_NVENC_AVAILABLE: Optional[bool] = None


def _encode_video(frame_dir: Path, output_path: Path, fps: int, use_gpu: bool) -> None:
    """Codifica frames a video usando NVENC si est? disponible y use_gpu=True"""
    gpu_info = gpu_utils.detect_cuda_gpu()
    if use_gpu:
        gpu_utils.force_cuda_gpu()
    
    if use_gpu and gpu_info["nvenc_available"]:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.png"),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",  # Preset balanceado calidad/velocidad
            "-rc:v",
            "vbr",  # Variable bitrate
            "-cq",
            "20",  # Calidad constante (0-51, menor es mejor)
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        if _try_encode(cmd):
            return
    
    # Fallback a libx264 con alta calidad
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",  # Calidad constante (0-51, menor es mejor)
        "-preset",
        "medium",  # Preset balanceado
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


@app.get("/status")
async def status(job_id: str):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job


@app.get("/health")
async def health():
    gpu_info = gpu_utils.detect_cuda_gpu()
    return {
        "status": "ok",
        "gpu_available": gpu_info["cuda_available"],
        "nvenc_available": gpu_info["nvenc_available"],
    }


def _set_job(job_id: str, total: int, output_path: str, job_type: str) -> None:
    with _JOB_LOCK:
        _JOBS[job_id] = {
            "type": job_type,
            "status": "queued",
            "progress": 0,
            "total": total,
            "message": "En cola",
            "log": ["En cola"],
            "output_path": output_path,
            "track_path": None,
            "error": None,
        }


def _update_job(job_id: str, **kwargs) -> None:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        if "log" in kwargs and isinstance(kwargs["log"], str):
            job["log"].append(kwargs["log"])
            if len(job["log"]) > 20:
                job["log"] = job["log"][-20:]
            kwargs = {k: v for k, v in kwargs.items() if k != "log"}
        job.update(kwargs)


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 0:
        return "calculando..."
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _render_task(job_id: str, config: RenderConfig, jobs, frame_dir: Path, output_path: Path) -> None:
    import time
    import shutil as shutil_mod
    
    try:
        if config.use_gpu:
            import gpu_utils
            gpu_utils.force_cuda_gpu()
            
        # If GPU is enabled, we MUST run in a single process/thread to maintain CUDA context
        if config.use_gpu and GPU_RENDER_AVAILABLE:
            workers = 1
            # Check GPU status
            gpu_stat = init_gpu()
            if not gpu_stat.get("available", False):
                _update_job(job_id, log=f"Advertencia: GPU no disponible ({gpu_stat.get('error')}), usando CPU.")
                config.use_gpu = False # Fallback to CPU
                # Restore worker count if fallback? No, simpler to stick to 1 or let it be.
                workers = config.workers if config.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        else:
            workers = config.workers if config.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

        _update_job(job_id, status="rendering", message="Renderizando frames", log="Renderizando frames")
        
        start_time = time.time()
        total_frames = len(jobs)
        
        # Frame cache to avoid re-rendering identical frames
        frame_cache = {}
        cache_precision = 4
        cache_hits = 0
        if workers <= 1:
            # OPTIMIZATION: Pre-load whole track area to GPU if available
            if config.use_gpu and GPU_RENDER_AVAILABLE:
                try:
                    from render_gpu import preload_track_gpu
                    _update_job(job_id, log="Optimizando texturas en GPU para máxima velocidad...")
                    preload_track_gpu(config, jobs)
                except Exception as e:
                    _update_job(job_id, log=f"Aviso: Falló precarga GPU ({e}). Cambiando a renderizado CPU multicore.")
                    config.use_gpu = False
                    # Recalculate workers for CPU mode
                    workers = config.workers if config.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

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
                for idx, center_e, center_n, heading, frame_path in jobs:
                    # Create cache key
                    cache_key = (round(center_e, cache_precision), round(center_n, cache_precision), round(heading, 1))
                    
                    if cache_key in frame_cache:
                        shutil_mod.copy2(frame_cache[cache_key], frame_path)
                        cache_hits += 1
                    else:
                        # Use a safe overscan for clipping vectors to avoid truncation during rotation
                        clip_margin = config.map_half_width_m * 2.0
                        bbox = (
                            center_e - clip_margin,
                            center_n - clip_margin,
                            center_e + clip_margin,
                            center_n + clip_margin,
                        )
                        clipped = clip_vectors(vectors, bbox)
                        # Dispatch rendering (handles GPU/CPU switch and fallback)
                        frame = _dispatch_render(config, dataset, clipped, center_e, center_n, heading, job_id=job_id, frame_idx=idx)

                        frame.save(frame_path, "PNG")
                        frame_cache[cache_key] = frame_path
                    
                    # Calculate ETA
                    # Calculate ETA
                    elapsed = time.time() - start_time
                    frames_done = idx + 1
                    
                    if frames_done > 0 and elapsed > 0:
                        fps_rate = frames_done / elapsed
                        remaining = total_frames - frames_done
                        eta_str = _format_eta(remaining / fps_rate if fps_rate > 0 else 0)
                        msg = f"Frame {frames_done}/{total_frames} • ETA: {eta_str}"
                    else:
                        msg = f"Frame {frames_done}/{total_frames}"
                    _update_job(job_id, progress=frames_done, message=msg)
        else:
            ctx = get_context("spawn")
            total = len(jobs)
            with ctx.Pool(
                processes=workers,
                initializer=init_worker,
                initargs=(
                    config.ortho_path,
                    [layer.model_dump() for layer in config.vector_layers],
                    config.vectors_paths,
                    config.curves_path,
                    config.line_color,
                    config.line_width,
                    config.boundary_color,
                    config.boundary_width,
                    config.point_color,
                    config.width,
                    config.height,
                    config.map_half_width_m,
                    config.arrow_size_px,
                    config.cone_angle_deg,
                    config.cone_length_px,
                    config.cone_opacity,
                    config.icon_circle_opacity,
                    config.icon_circle_size_px,
                    config.show_compass,
                    config.compass_size_px,
                ),
            ) as pool:
                done = 0
                for _ in pool.imap_unordered(
                    render_frame_job, jobs, chunksize=max(1, total // (workers * 4))
                ):
                    done += 1
                    elapsed = time.time() - start_time
                    if done > 0 and elapsed > 0:
                        fps_rate = done / elapsed
                        remaining = total - done
                        eta_str = _format_eta(remaining / fps_rate if fps_rate > 0 else 0)
                        msg = f"Frame {done}/{total} • ETA: {eta_str}"
                    else:
                        msg = f"Frame {done}/{total}"
                    _update_job(job_id, progress=done, message=msg)

        if cache_hits > 0:
            cache_pct = (cache_hits / total_frames) * 100
            _update_job(job_id, log=f"Cache: {cache_hits} frames reutilizados ({cache_pct:.1f}%)")
        
        _update_job(job_id, status="encoding", message="Codificando video", log="Codificando video")
        _encode_video(frame_dir, output_path, config.fps, config.use_gpu)
        
        total_time = time.time() - start_time
        _update_job(job_id, status="finished", progress=len(jobs), message=f"Finalizado en {_format_eta(total_time)}", log=f"Tiempo total: {_format_eta(total_time)}")
    except Exception as exc:
        _update_job(job_id, status="error", message="Error en render", log=str(exc), error=str(exc))
    finally:
        if GPU_RENDER_AVAILABLE:
            # Liberar memoria GPU al terminar (éxito o error)
            cleanup_gpu()


def _track_task(job_id: str, req: TrackRequest, output_path: Path) -> None:
    try:
        _update_job(job_id, status="tracking", message="Analizando video", log="Analizando video")

        def _progress(done: int, total: int):
            _update_job(job_id, progress=done, total=total, message=f"Frame {done}/{total}")

        track_points(
            req.video_path,
            [point.model_dump() for point in req.points],
            str(output_path),
            progress_cb=_progress,
        )
        _update_job(
            job_id,
            status="finished",
            message="Tracking finalizado",
            log="Tracking finalizado",
            track_path=str(output_path),
        )
    except Exception as exc:
        _update_job(job_id, status="error", message="Error en tracking", log=str(exc), error=str(exc))


def _render_overlay_task(job_id: str, req: OverlayRequest, output_path: Path) -> None:
    try:
        _update_job(job_id, status="rendering", message="Renderizando overlay", log="Renderizando overlay")

        def _progress(done: int, total: int):
            _update_job(job_id, progress=done, total=total, message=f"Frame {done}/{total}")

        render_overlay(
            req.video_path,
            req.track_path,
            str(output_path),
            req.line_color,
            req.line_width,
            req.point_color,
            req.show_points,
            req.snap_to_edges,
            req.snap_radius,
            req.smooth_alpha,
            req.max_jump,
            progress_cb=_progress,
        )
        _update_job(job_id, status="finished", message="Overlay finalizado", log="Overlay finalizado")
    except Exception as exc:
        _update_job(job_id, status="error", message="Error en overlay", log=str(exc), error=str(exc))


def _try_encode(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _nvenc_available() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in result.stdout
        return _NVENC_AVAILABLE
    except Exception:
        _NVENC_AVAILABLE = False
        return False
