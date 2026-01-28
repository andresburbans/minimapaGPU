import atexit
import math
import io
import urllib.request
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from shapely.errors import GEOSException
from shapely.geometry import LineString, Point, Polygon, box
from PIL import Image, ImageDraw


@dataclass
class Segment:
    t_start: float
    t_end: float
    e_start: float
    n_start: float
    e_end: float
    n_end: float
    heading_start: Optional[float]
    heading_end: Optional[float]


def load_segments(csv_path: str) -> List[Segment]:
    df = pd.read_csv(csv_path)
    required = {"t_start", "t_end", "E_ini", "N_ini", "E_fin", "N_fin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    heading_start = df["heading_ini"] if "heading_ini" in df.columns else None
    heading_end = df["heading_fin"] if "heading_fin" in df.columns else None
    heading_single = df["heading"] if "heading" in df.columns else None
    segments: List[Segment] = []
    for idx, row in df.iterrows():
        h_start = None
        h_end = None
        if heading_single is not None:
            h_start = float(row["heading"])
            h_end = float(row["heading"])
        else:
            if heading_start is not None:
                h_start = float(row["heading_ini"])
            if heading_end is not None:
                h_end = float(row["heading_fin"])
        segments.append(
            Segment(
                t_start=float(row["t_start"]),
                t_end=float(row["t_end"]),
                e_start=float(row["E_ini"]),
                n_start=float(row["N_ini"]),
                e_end=float(row["E_fin"]),
                n_end=float(row["N_fin"]),
                heading_start=h_start,
                heading_end=h_end,
            )
        )
    return segments


def interpolate_position(segments: List[Segment], t: float) -> Tuple[float, float, float]:
    """Interpolate position and heading at time t.
    
    Handles:
    - Times within a segment: linear interpolation
    - Times before first segment: use first segment start position
    - Times after last segment: use last segment end position
    - Times in gaps between segments: interpolate between adjacent segment end/start
    """
    if not segments:
        return 0.0, 0.0, 0.0
    
    # Sort segments by start time to ensure correct ordering
    sorted_segments = sorted(segments, key=lambda s: s.t_start)
    
    # Before first segment
    first = sorted_segments[0]
    if t < first.t_start:
        heading = first.heading_start if first.heading_start is not None else compute_heading(
            first.e_start, first.n_start, first.e_end, first.n_end
        )
        return first.e_start, first.n_start, heading
    
    # Check each segment and gaps between them
    for i, seg in enumerate(sorted_segments):
        # Within this segment
        if seg.t_start <= t <= seg.t_end:
            span = max(seg.t_end - seg.t_start, 1e-6)
            ratio = (t - seg.t_start) / span
            e = seg.e_start + (seg.e_end - seg.e_start) * ratio
            n = seg.n_start + (seg.n_end - seg.n_start) * ratio
            if seg.heading_start is not None and seg.heading_end is not None:
                heading = seg.heading_start + (seg.heading_end - seg.heading_start) * ratio
            else:
                heading = compute_heading(seg.e_start, seg.n_start, seg.e_end, seg.n_end)
            return e, n, heading
        
        # Check if we're in a gap before the next segment
        if i < len(sorted_segments) - 1:
            next_seg = sorted_segments[i + 1]
            if seg.t_end < t < next_seg.t_start:
                # Interpolate through the gap between segments
                gap_duration = next_seg.t_start - seg.t_end
                gap_ratio = (t - seg.t_end) / max(gap_duration, 1e-6)
                
                # Linear interpolation for position (removed easing/smoothing)
                eased_ratio = gap_ratio
                
                # Interpolate position from end of current segment to start of next
                e = seg.e_end + (next_seg.e_start - seg.e_end) * eased_ratio
                n = seg.n_end + (next_seg.n_start - seg.n_end) * eased_ratio
                
                # Interpolate heading with easing
                h_end = seg.heading_end if seg.heading_end is not None else compute_heading(
                    seg.e_start, seg.n_start, seg.e_end, seg.n_end
                )
                h_next_start = next_seg.heading_start if next_seg.heading_start is not None else compute_heading(
                    next_seg.e_start, next_seg.n_start, next_seg.e_end, next_seg.n_end
                )
                heading = interpolate_heading_smooth(h_end, h_next_start, gap_ratio, use_easing=False)
                
                return e, n, heading
    
    # After last segment
    last = sorted_segments[-1]
    heading = last.heading_end if last.heading_end is not None else compute_heading(
        last.e_start, last.n_start, last.e_end, last.n_end
    )
    return last.e_end, last.n_end, heading


def compute_heading(e1: float, n1: float, e2: float, n2: float) -> float:
    dx = e2 - e1
    dy = n2 - n1
    angle_rad = math.atan2(dx, dy)
    return (math.degrees(angle_rad) + 360.0) % 360.0


def ease_in_out_cubic(t: float) -> float:
    """Cubic easing function for smooth acceleration/deceleration.
    
    Args:
        t: Value between 0 and 1
    Returns:
        Eased value between 0 and 1
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_out_quad(t: float) -> float:
    """Quadratic easing function (lighter than cubic).
    
    Args:
        t: Value between 0 and 1
    Returns:
        Eased value between 0 and 1
    """
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - pow(-2 * t + 2, 2) / 2


def interpolate_heading_smooth(h_start: float, h_end: float, ratio: float, use_easing: bool = True) -> float:
    """Interpolate between two headings with optional easing and wrap-around handling.
    
    Args:
        h_start: Starting heading in degrees (0-360)
        h_end: Ending heading in degrees (0-360)
        ratio: Interpolation ratio (0-1)
        use_easing: Whether to apply easing function
    
    Returns:
        Interpolated heading in degrees (0-360)
    """
    # Apply easing to the ratio
    # Easing disabled per user request to avoid deformation/smoothing artifacts
    if use_easing:
        # ratio = ease_in_out_quad(ratio)
        pass # Stay linear
    
    # Handle wrap-around (e.g., 350┬░ to 10┬░ should go through 0┬░)
    h_diff = h_end - h_start
    if h_diff > 180:
        h_diff -= 360
    elif h_diff < -180:
        h_diff += 360
    
    return (h_start + h_diff * ratio + 360) % 360


def _to_rgba(image: np.ndarray, nodata_val=None) -> np.ndarray:
    """Convierte data rasterio (b, h, w) a (h, w, 4) manteniendo nodata como transparente"""
    count, h, w = image.shape
    
    # 1. Extraer color
    if count >= 3:
        rgb = image[:3] # (3, h, w)
    else:
        single = image[0]
        rgb = np.stack([single, single, single]) # (3, h, w)
        
    rgb = np.transpose(rgb, (1, 2, 0)) # (h, w, 3)
    
    # 2. Crear Mascara Alpha
    if count == 4:
        # Ya tiene alpha
        alpha = image[3] # (h, w)
    elif nodata_val is not None:
        # Usar nodata del primer canal para mascara
        # Asumimos que si un canal es nodata, es transparente
        # Cuidado con floats vs ints
        if np.isnan(nodata_val):
             mask = ~np.isnan(image[0])
        else:
             mask = image[0] != nodata_val
        alpha = (mask.astype(np.uint8) * 255)
    else:
        # Opaco total
        alpha = np.full((h, w), 255, dtype=np.uint8)
        
    return rgb, alpha


def _normalize_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Normaliza data RGB (float/int) a uin8 y combina con alpha presuave"""
    # Normalize RGB
    if rgb.dtype != np.uint8:
        img_f = rgb.astype(np.float32)
        valid_mask = alpha > 0
        if np.any(valid_mask):
            # Calcular min/max solo en areas visibles
            # Para evitar que el negro de fondo distorsione el contraste
            # flattened visible vals
            vals = img_f[valid_mask]
            min_val = np.nanmin(vals)
            max_val = np.nanmax(vals)
            if max_val - min_val < 1e-6:
                rgb_u8 = np.zeros_like(img_f, dtype=np.uint8)
            else:
                 scaled = (img_f - min_val) / (max_val - min_val)
                 rgb_u8 = (scaled * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb_u8 = np.zeros(rgb.shape, dtype=np.uint8)
    else:
        rgb_u8 = rgb

    return np.dstack((rgb_u8, alpha))


_GLOBAL_NAV_ICON: Optional[Image.Image] = None
_NAV_ICON_PATH = Path(__file__).resolve().parents[1] / "web" / "public" / "navIcon.png"


def _load_nav_icon() -> Optional[Image.Image]:
    global _GLOBAL_NAV_ICON
    if _GLOBAL_NAV_ICON is not None:
        return _GLOBAL_NAV_ICON
    try:
        _GLOBAL_NAV_ICON = Image.open(_NAV_ICON_PATH).convert("RGBA")
    except Exception as exc:
        print(f"[icon] No se pudo cargar navIcon.png: {exc}")
        _GLOBAL_NAV_ICON = None
    return _GLOBAL_NAV_ICON


def render_frame(
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
    """Render a single minimap frame.
    
    The map rotates so that the direction of travel points DOWNWARD (south on screen),
    giving the sensation that the map moves from top to bottom as we advance.
    The navigation icon and radar cone always point UP (north on screen).
    """
    res_x, res_y = dataset.res
    overscan = max(1.0, _ROTATION_OVERSCAN)
    
    # To support rotation without black corners and without aspect ratio distortion,
    # we first define the geographic square that encompasses the diagonal of our view.
    
    # Supersampling factor to eliminate flickering (render larger, then scale down)
    ss_factor = 2
    
    # 1. Calculate how many meters are represented by one pixel of the output width
    meters_per_pixel = (map_half_width_m * 2.0) / width
    
    # 2. Determine the pixel diagonal of the output frame
    diag_px = math.sqrt(width**2 + height**2)
    # We add a 15% margin to ensure no corners appear even with slight floating point errors
    render_size_px = int(diag_px * 1.15)
    
    # 3. Apply supersampling to the render size
    ss_render_size_px = render_size_px * ss_factor
    
    # 4. Define the geographic extent of our square render base (in meters)
    render_size_m = render_size_px * meters_per_pixel
    
    # 5. Strictly compute the geographical bounds for this square
    xmin = center_e - render_size_m / 2
    xmax = center_e + render_size_m / 2
    ymin = center_n - render_size_m / 2
    ymax = center_n + render_size_m / 2
    
    # 6. Read the raster data at supersampled resolution
    window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, dataset.transform)
    
    data = dataset.read(
        window=window,
        out_shape=(dataset.count, ss_render_size_px, ss_render_size_px),
        resampling=Resampling.bilinear,
        boundless=True,
        fill_value=dataset.nodata # Importante para detectar areas fuera del raster
    )
    
    # Procesamiento con transparencia
    rgb_raw, alpha_raw = _to_rgba(data, nodata_val=dataset.nodata)
    rgba = _normalize_rgba(rgb_raw, alpha_raw)
    base = Image.fromarray(rgba, mode="RGBA")

    # 7. Use supersampled mapping for vector drawing
    draw = ImageDraw.Draw(base, "RGBA")
    def map_to_px(e, n):
        px = (e - xmin) / (xmax - xmin) * ss_render_size_px
        py = (ymax - n) / (ymax - ymin) * ss_render_size_px
        return px, py

    for geom_iter, color, line_width, pattern in vectors:
        for geom in geom_iter:
            # Multiply line_width by ss_factor to maintain scale
            _draw_geometry_precise(draw, geom, map_to_px, color, int(line_width * ss_factor), pattern)

    # 8. Downsample to the intended render size (this provides high-quality anti-aliasing)
    # Use Image.LANCZOS for the best reduction quality
    base = base.resize((render_size_px, render_size_px), resample=Image.LANCZOS)

    # 9. Rotate the map so that direction of travel (destination B) always points UP.
    rotation_angle = heading
    rotated = _rotate_image(base, rotation_angle, target_size=(width, height))
    
    center_px = (width // 2, height // 2)
    
    # In this mode, the icon and radar always point forward (UP)
    # because the world rotates around the vehicle.
    _draw_cone(rotated, center_px, 0.0, cone_angle_deg, cone_length_px, cone_opacity)
    _draw_center_icon(
        rotated,
        center_px,
        arrow_size_px,
        icon_circle_opacity,
        icon_circle_size_px,
        0.0,  # Icon always points UP
    )
    
    # Draw compass showing true north
    # If the map is rotated by 'heading' (CCW), North is at angle -heading (CW) from the top.
    if show_compass:
        compass_size = max(15, compass_size_px)
        compass_margin = max(10, compass_size // 2)
        compass_pos = (width - compass_margin - compass_size, compass_margin + compass_size)
        _draw_compass(rotated, compass_pos, compass_size, -heading)
    
    return rotated


def _draw_compass(img: Image.Image, center: Tuple[int, int], size: int, heading: float) -> None:
    """Draw a compass indicator showing true north direction.
    
    The compass rotates to show where geographic north is relative to the current view.
    If the map is rotated by ╬©, North (originally up) is now at -╬©.
    """
    cx, cy = center
    # North is at 'heading' angle from current vertical UP
    north_angle_rad = math.radians(heading)
    
    # Draw subtle background circle
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    bg_radius = int(size * 1.1)
    draw.ellipse(
        (cx - bg_radius, cy - bg_radius, cx + bg_radius, cy + bg_radius),
        fill=(0, 0, 0, 80)
    )
    img.alpha_composite(overlay)
    
    # Draw north arrow (red)
    north_tip = (
        cx + size * math.sin(north_angle_rad),
        cy - size * math.cos(north_angle_rad)
    )
    north_base_l = (
        cx + size * 0.3 * math.sin(north_angle_rad + math.radians(140)),
        cy - size * 0.3 * math.cos(north_angle_rad + math.radians(140))
    )
    north_base_r = (
        cx + size * 0.3 * math.sin(north_angle_rad - math.radians(140)),
        cy - size * 0.3 * math.cos(north_angle_rad - math.radians(140))
    )
    
    draw2 = ImageDraw.Draw(img, "RGBA")
    draw2.polygon([north_tip, north_base_l, (cx, cy), north_base_r], fill=(220, 60, 60, 230))
    
    # Draw south arrow (white/gray)
    south_angle_rad = north_angle_rad + math.pi
    south_tip = (
        cx + size * 0.6 * math.sin(south_angle_rad),
        cy - size * 0.6 * math.cos(south_angle_rad)
    )
    south_base_l = (
        cx + size * 0.25 * math.sin(south_angle_rad + math.radians(140)),
        cy - size * 0.25 * math.cos(south_angle_rad + math.radians(140))
    )
    south_base_r = (
        cx + size * 0.25 * math.sin(south_angle_rad - math.radians(140)),
        cy - size * 0.25 * math.cos(south_angle_rad - math.radians(140))
    )
    draw2.polygon([south_tip, south_base_l, (cx, cy), south_base_r], fill=(200, 200, 200, 180))
    
    # Draw "N" label near north arrow
    try:
        from PIL import ImageFont
        font_size = max(8, size // 3)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        n_pos = (
            cx + (size + font_size) * math.sin(north_angle_rad) - font_size // 2,
            cy - (size + font_size) * math.cos(north_angle_rad) - font_size // 2
        )
        draw2.text(n_pos, "N", fill=(255, 80, 80, 255), font=font)
    except:
        pass  # Skip text if font not available


def _draw_geometry_precise(
    draw: ImageDraw.ImageDraw, geom, map_func, color: str, line_width: int, pattern: str
) -> None:
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        coords = [map_func(pt[0], pt[1]) for pt in geom.coords]
        _draw_path(draw, coords, color, line_width, pattern)
    elif isinstance(geom, Polygon):
        exterior = [map_func(pt[0], pt[1]) for pt in geom.exterior.coords]
        _draw_path(draw, exterior, color, line_width, pattern)
        for interior in geom.interiors:
            interior_coords = [map_func(pt[0], pt[1]) for pt in interior.coords]
            _draw_path(draw, interior_coords, color, line_width, pattern)
    elif isinstance(geom, Point):
        x, y = map_func(geom.x, geom.y)
        r = max(line_width, 3)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    else:
        if hasattr(geom, "geoms"):
            for sub in geom.geoms:
                _draw_geometry_precise(draw, sub, map_func, color, line_width, pattern)


def _draw_path(
    draw: ImageDraw.ImageDraw,
    coords: List[Tuple[float, float]],
    color: str,
    line_width: int,
    pattern: str,
) -> None:
    if len(coords) < 2:
        return
    pattern = (pattern or "solid").lower()
    if pattern == "solid":
        draw.line(coords, fill=color, width=line_width)
        return
    if pattern == "dashed":
        _draw_dashed_line(draw, coords, color, line_width, [12, 8])
        return
    if pattern == "dotted":
        _draw_dotted_line(draw, coords, color, line_width, max(6, line_width * 3))
        return
    if pattern == "dashdot":
        _draw_dashed_line(draw, coords, color, line_width, [14, 6, 3, 6])
        return
    if pattern == "tactical":
        _draw_dashed_line(draw, coords, color, line_width, [16, 6, 2, 6])
        return
    if pattern == "road-arrows":
        draw.line(coords, fill=color, width=line_width)
        _draw_road_arrows(draw, coords, color, line_width)
        return
    draw.line(coords, fill=color, width=line_width)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    coords: List[Tuple[float, float]],
    color: str,
    width: int,
    pattern: List[float],
) -> None:
    if len(coords) < 2:
        return
    pattern_index = 0
    pattern_pos = 0.0
    draw_on = True
    pattern_len = pattern[pattern_index]
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 0:
            continue
        dx = (x2 - x1) / seg_len
        dy = (y2 - y1) / seg_len
        dist = 0.0
        while dist < seg_len:
            remaining = pattern_len - pattern_pos
            step = min(seg_len - dist, remaining)
            x_start = x1 + dx * dist
            y_start = y1 + dy * dist
            x_end = x1 + dx * (dist + step)
            y_end = y1 + dy * (dist + step)
            if draw_on:
                draw.line([(x_start, y_start), (x_end, y_end)], fill=color, width=width)
            dist += step
            pattern_pos += step
            if pattern_pos >= pattern_len - 1e-6:
                pattern_index = (pattern_index + 1) % len(pattern)
                pattern_len = pattern[pattern_index]
                pattern_pos = 0.0
                draw_on = pattern_index % 2 == 0


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    coords: List[Tuple[float, float]],
    color: str,
    width: int,
    spacing: float,
) -> None:
    radius = max(1.5, width / 2)
    for x, y, _angle in _iter_polyline_positions(coords, spacing, spacing * 0.5):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _draw_road_arrows(
    draw: ImageDraw.ImageDraw,
    coords: List[Tuple[float, float]],
    color: str,
    width: int,
) -> None:
    arrow_spacing = max(width * 6, 32)
    arrow_size = max(width * 2.4, 8)
    arrow_color = _lighten_color(color, 0.25)
    for x, y, angle in _iter_polyline_positions(coords, arrow_spacing, arrow_spacing * 0.5):
        _draw_arrow_marker(draw, x, y, angle, arrow_size, arrow_color)


def _draw_arrow_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    angle_deg: float,
    size: float,
    color: str,
) -> None:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cos_p = -sin_a
    sin_p = cos_a
    tip = (x + cos_a * size, y + sin_a * size)
    base = (x - cos_a * size * 0.5, y - sin_a * size * 0.5)
    left = (base[0] + cos_p * size * 0.5, base[1] + sin_p * size * 0.5)
    right = (base[0] - cos_p * size * 0.5, base[1] - sin_p * size * 0.5)
    draw.polygon([tip, left, right], fill=color)


def _iter_polyline_positions(
    coords: List[Tuple[float, float]],
    spacing: float,
    offset: float = 0.0,
):
    if len(coords) < 2:
        return
    next_dist = offset if offset > 0 else spacing
    total_dist = 0.0
    x_prev, y_prev = coords[0]
    for i in range(1, len(coords)):
        x_curr, y_curr = coords[i]
        seg_len = math.hypot(x_curr - x_prev, y_curr - y_prev)
        if seg_len <= 0:
            continue
        while total_dist + seg_len >= next_dist:
            step = next_dist - total_dist
            t = step / seg_len
            x = x_prev + (x_curr - x_prev) * t
            y = y_prev + (y_curr - y_prev) * t
            angle = math.degrees(math.atan2(y_curr - y_prev, x_curr - x_prev))
            yield x, y, angle
            next_dist += spacing
        total_dist += seg_len
        x_prev, y_prev = x_curr, y_curr


def _lighten_color(color: str, amount: float) -> str:
    rgb = _parse_hex_color(color)
    if rgb is None:
        return color
    r, g, b = rgb
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _parse_hex_color(color: str) -> Optional[Tuple[int, int, int]]:
    if not color:
        return None
    value = color.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except ValueError:
        return None
    return r, g, b


    return r, g, b


def _clamp_latlon(lat: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, lat))


def _latlon_to_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    lat = _clamp_latlon(lat, -85.05112878, 85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * 256.0
    return x, y


def _pixel_to_latlon(x: float, y: float, zoom: int) -> Tuple[float, float]:
    n = 2 ** zoom
    lon_deg = x / (n * 256.0) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / (n * 256.0))))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


@functools.lru_cache(maxsize=1000)
def _fetch_tile(x: int, y: int, z: int):
    # Intentar descargar tile satelital
    url = f"https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MinimapaGPT/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        # Retornar tile negro/vac├¡o si falla
        return Image.new("RGB", (256, 256), (20, 20, 20))


def _fetch_wms_mosaic_for_bounds(west: float, south: float, east: float, north: float, zoom: int) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """Descarga mosaico WMS y devuelve la imagen y su bounding box real (px_west, px_north, px_east, px_south) en coords pixel del zoom global"""
    px_west_f, px_north_f = _latlon_to_pixel(north, west, zoom)
    px_east_f, px_south_f = _latlon_to_pixel(south, east, zoom)
    
    min_tx = int(px_west_f // 256)
    max_tx = int(px_east_f // 256)
    min_ty = int(px_north_f // 256)
    max_ty = int(px_south_f // 256)

    tiles_x = max_tx - min_tx + 1
    tiles_y = max_ty - min_ty + 1
    
    # Cap tiles if too large
    if tiles_x * tiles_y > 100:
        return Image.new("RGB", (100, 100), (0,0,0)), (0,0)

    mosaic = Image.new("RGB", (tiles_x * 256, tiles_y * 256), (20, 20, 20))
    for ty in range(min_ty, max_ty + 1):
        for tx in range(min_tx, max_tx + 1):
            tile = _fetch_tile(tx, ty, zoom)
            mosaic.paste(tile, ((tx - min_tx) * 256, (ty - min_ty) * 256))
            
    # Bounding box del mosaico completo en coordenadas p├¡xel globales
    mosaic_px_left = min_tx * 256
    mosaic_px_top = min_ty * 256
    # Width/Height
    
    return mosaic, (mosaic_px_left, mosaic_px_top)


def _map_to_pixel(pt: Tuple[float, float], transform) -> Tuple[float, float]:
    # Shapely puede entregar coordenadas 3D (x, y, z); usamos solo x/y.
    x = pt[0]
    y = pt[1]
    col, row = ~transform * (x, y)
    return float(col), float(row)


def _draw_arrow(img: Image.Image, center: Tuple[int, int], heading: float, size: int) -> None:
    angle = math.radians(heading)
    cx, cy = center
    tip = (cx + size * math.sin(angle), cy - size * math.cos(angle))
    left = (cx + size * 0.6 * math.sin(angle + math.radians(150)),
            cy - size * 0.6 * math.cos(angle + math.radians(150)))
    right = (cx + size * 0.6 * math.sin(angle - math.radians(150)),
             cy - size * 0.6 * math.cos(angle - math.radians(150)))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.polygon([tip, left, right], fill="#ffffff", outline="#0d1016")


def _rotate_image(
    img: Image.Image,
    angle_deg: float,
    target_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    if abs(angle_deg) < 0.01:
        if target_size is None:
            return img
        w, h = img.size
        target_w, target_h = target_size
        left = int((w - target_w) / 2)
        top = int((h - target_h) / 2)
        return img.crop((left, top, left + target_w, top + target_h))
    
    # Use Image.BICUBIC for widest compatibility with the rotate method
    # Expand=True adds transparent padding if mode is RGBA
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    target_w, target_h = target_size if target_size else img.size
    rw, rh = rotated.size
    left = int((rw - target_w) / 2)
    top = int((rh - target_h) / 2)
    return rotated.crop((left, top, left + target_w, top + target_h))


def _draw_center_icon(
    img: Image.Image,
    center: Tuple[int, int],
    size: int,
    circle_opacity: float,
    circle_size_px: int,
    heading: float = 0.0,
) -> None:
    """Draw the navigation icon at center, rotated to point in the heading direction.
    
    Args:
        heading: Direction in degrees (0=north/up, 90=east/right, 180=south/down, 270=west/left)
    """
    cx, cy = center
    icon_size = max(12, int(size))
    circle_alpha = int(255 * max(0.0, min(1.0, circle_opacity)))
    if circle_alpha > 0:
        circle_diameter = max(16, int(circle_size_px))
        circle_r = max(8, int(circle_diameter / 2))
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        bbox = (cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r)
        draw.ellipse(bbox, fill=(255, 255, 255, circle_alpha))
        img.alpha_composite(overlay)

    icon = _load_nav_icon()
    if icon is None:
        _draw_arrow(img, center, heading, icon_size)
        return

    w, h = icon.size
    scale = icon_size / max(w, h, 1)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    icon_scaled = icon.resize((new_w, new_h), resample=Image.LANCZOS)
    
    # Rotate the icon to point in the heading direction
    # PIL rotates counter-clockwise, heading is clockwise from north
    # So we need to rotate by -heading
    if heading != 0:
        # Increase size for rotation to avoid clipping
        diagonal = int(math.sqrt(new_w**2 + new_h**2))
        padded = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))
        paste_x = (diagonal - new_w) // 2
        paste_y = (diagonal - new_h) // 2
        padded.paste(icon_scaled, (paste_x, paste_y))
        # Rotate (PIL rotates counter-clockwise, so use -heading for clockwise rotation)
        icon_rotated = padded.rotate(-heading, resample=Image.BICUBIC, expand=False)
        icon_scaled = icon_rotated
        new_w, new_h = icon_scaled.size
    
    left = int(cx - new_w / 2)
    top = int(cy - new_h / 2)
    img.alpha_composite(icon_scaled, dest=(left, top))


def _draw_cone(img: Image.Image, center: Tuple[int, int], heading: float, angle_deg: float, length: int, opacity: float) -> None:
    """Draw a directional cone/radar pointing in the heading direction.
    
    In PIL's pieslice, angles are measured:
    - 0┬░ = right (3 o'clock) = east
    - 90┬░ = down (6 o'clock) = south
    - 180┬░ = left (9 o'clock) = west
    - 270┬░ / -90┬░ = up (12 o'clock) = north
    
    Geographic heading:
    - 0┬░ = north (up)
    - 90┬░ = east (right)
    - 180┬░ = south (down)
    - 270┬░ = west (left)
    
    Conversion: pil_angle = -90 + heading
    """
    if opacity <= 0 or length <= 0 or angle_deg <= 0:
        return
    cx, cy = center
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    # Convert geographic heading to PIL angle
    # -90 (or 270) points UP in PIL, heading=0 is north/up
    pil_angle = -90 + heading
    start = pil_angle - angle_deg / 2
    end = pil_angle + angle_deg / 2
    steps = 6
    base_alpha = max(0.04, min(0.4, opacity)) * 0.6
    for i in range(steps, 0, -1):
        radius = int(length * (i / steps))
        falloff = (i / steps) ** 1.4
        alpha = int(255 * base_alpha * falloff)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.pieslice(bbox, start=start, end=end, fill=(235, 235, 235, alpha))
    img.alpha_composite(overlay)


def load_vectors(
    dataset_crs,
    vector_layers: List[dict],
    vectors_paths: List[str],
    curves_path: Optional[str],
    line_color: str,
    line_width: int,
    boundary_color: str,
    boundary_width: int,
    point_color: str,
) -> List[Tuple[Iterable, str, int, str]]:
    layers: List[Tuple[Iterable, str, int, str]] = []
    if not vectors_paths and not curves_path and not vector_layers:
        return layers

    def read_path(path: str):
        import geopandas as gpd
        from pyproj import CRS

        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            print(f"[vectors] No se pudo leer {path}: {exc}")
            return []
        if gdf.empty:
            return []
        if "geometry" not in gdf:
            return []
        gdf = gdf[gdf.geometry.notnull()]
        try:
            gdf = gdf[~gdf.geometry.is_empty]
        except Exception:
            pass
        if gdf.empty:
            return []
        dataset_crs_obj = CRS.from_user_input(dataset_crs)
        if gdf.crs is None:
            minx, miny, maxx, maxy = gdf.total_bounds
            looks_like_lonlat = (
                -180.0 <= minx <= 180.0
                and -180.0 <= maxx <= 180.0
                and -90.0 <= miny <= 90.0
                and -90.0 <= maxy <= 90.0
            )
            # Muchos GeoJSON vienen sin CRS pero en EPSG:4326 (lon/lat).
            if looks_like_lonlat and not dataset_crs_obj.is_geographic:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.set_crs(dataset_crs_obj)
        if CRS.from_user_input(gdf.crs) != dataset_crs_obj:
            gdf = gdf.to_crs(dataset_crs_obj)
        return gdf.geometry

    if vector_layers:
        for layer in vector_layers:
            layers.append(
                (
                    read_path(layer["path"]),
                    layer.get("color", line_color),
                    int(layer.get("width", line_width)),
                    layer.get("pattern", "solid"),
                )
            )
    else:
        for path in vectors_paths:
            layers.append((read_path(path), line_color, line_width, "solid"))
    if curves_path:
        layers.append((read_path(curves_path), boundary_color, boundary_width, "tactical"))
    return layers


def clip_vectors(
    vectors: List[Tuple[Iterable, str, int, str]],
    bbox: Tuple[float, float, float, float],
) -> List[Tuple[Iterable, str, int, str]]:
    xmin, ymin, xmax, ymax = bbox
    clip_box = box(xmin, ymin, xmax, ymax)
    clipped = []
    for geoms, color, width, pattern in vectors:
        clipped_geoms = []
        for geom in geoms:
            if geom is None:
                continue
            try:
                if geom.is_empty:
                    continue
            except Exception:
                continue
            try:
                inter = geom.intersection(clip_box)
            except GEOSException:
                continue
            except Exception:
                continue
            if inter is None or inter.is_empty:
                continue
            clipped_geoms.append(inter)
        clipped.append((clipped_geoms, color, width, pattern))
    return clipped


_GLOBAL_DATASET = None
_GLOBAL_VECTORS = None
_GLOBAL_SETTINGS = None

_ROTATION_OVERSCAN = 1.45


def init_worker(
    ortho_path: str,
    vector_layers: List[dict],
    vectors_paths: List[str],
    curves_path: Optional[str],
    line_color: str,
    line_width: int,
    boundary_color: str,
    boundary_width: int,
    point_color: str,
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
) -> None:
    global _GLOBAL_DATASET, _GLOBAL_VECTORS, _GLOBAL_SETTINGS
    _GLOBAL_DATASET = rasterio.open(ortho_path)
    _load_nav_icon()
    _GLOBAL_VECTORS = load_vectors(
        _GLOBAL_DATASET.crs,
        vector_layers,
        vectors_paths,
        curves_path,
        line_color,
        line_width,
        boundary_color,
        boundary_width,
        point_color,
    )
    _GLOBAL_SETTINGS = {
        "width": width,
        "height": height,
        "map_half_width_m": map_half_width_m,
        "arrow_size_px": arrow_size_px,
        "cone_angle_deg": cone_angle_deg,
        "cone_length_px": cone_length_px,
        "cone_opacity": cone_opacity,
        "icon_circle_opacity": icon_circle_opacity,
        "icon_circle_size_px": icon_circle_size_px,
        "show_compass": show_compass,
        "compass_size_px": compass_size_px,
    }

    def _close_dataset():
        if _GLOBAL_DATASET is not None:
            _GLOBAL_DATASET.close()

    atexit.register(_close_dataset)


def render_frame_job(job: Tuple[int, float, float, float, str]) -> int:
    """
    Render a single frame with WMS background and orthomosaic overlay.
    
    IMPORTANT: Uses exactly the same geometry calculations as render_frame()
    to ensure pixel-perfect alignment between WMS and orthomosaic layers.
    """
    idx, center_e, center_n, heading, frame_path = job
    width = _GLOBAL_SETTINGS["width"]
    height = _GLOBAL_SETTINGS["height"]
    map_half_width_m = _GLOBAL_SETTINGS["map_half_width_m"]
    arrow_size_px = _GLOBAL_SETTINGS["arrow_size_px"]
    cone_angle_deg = _GLOBAL_SETTINGS["cone_angle_deg"]
    cone_length_px = _GLOBAL_SETTINGS["cone_length_px"]
    cone_opacity = _GLOBAL_SETTINGS["cone_opacity"]
    icon_circle_opacity = _GLOBAL_SETTINGS["icon_circle_opacity"]
    icon_circle_size_px = _GLOBAL_SETTINGS["icon_circle_size_px"]
    show_compass = _GLOBAL_SETTINGS.get("show_compass", True)
    compass_size_px = _GLOBAL_SETTINGS.get("compass_size_px", 40)

    # === CRITICAL: Use EXACTLY the same geometry as render_frame() ===
    # This ensures WMS and ortho layers are perfectly aligned
    meters_per_pixel = (map_half_width_m * 2.0) / width
    diag_px = math.sqrt(width**2 + height**2)
    render_size_px = int(diag_px * 1.15)  # Same 15% margin as render_frame
    render_size_m = render_size_px * meters_per_pixel

    # Use a safe margin for clipping vectors
    clip_margin = map_half_width_m * 2.0
    bbox = (
        center_e - clip_margin,
        center_n - clip_margin,
        center_e + clip_margin,
        center_n + clip_margin,
    )
    
    # === 1. Render WMS Base ===
    try:
        w_geo, s_geo, e_geo, n_geo = transform_bounds(
            _GLOBAL_DATASET.crs, 
            "EPSG:4326", 
            bbox[0], bbox[1], bbox[2], bbox[3], 
            densify_pts=21
        )
        zoom = 19
        
        mosaic_img, (mosaic_left_glob, mosaic_top_glob) = _fetch_wms_mosaic_for_bounds(w_geo, s_geo, e_geo, n_geo, zoom)
        
        # Calculate center relative to mosaic
        center_lon_geo = (w_geo + e_geo) / 2
        center_lat_geo = (s_geo + n_geo) / 2
        cx_glob, cy_glob = _latlon_to_pixel(center_lat_geo, center_lon_geo, zoom)
        
        rel_cx = cx_glob - mosaic_left_glob
        rel_cy = cy_glob - mosaic_top_glob
        
        # === CRITICAL: Use render_size_px for crop, same as ortho ===
        # This ensures both layers have identical pre-rotation geometry
        crop_size = render_size_px
        crop_l = int(rel_cx - crop_size / 2)
        crop_t = int(rel_cy - crop_size / 2)
        crop_r = crop_l + crop_size
        crop_b = crop_t + crop_size
        
        # Handle edge cases where mosaic might be smaller
        ms_w, ms_h = mosaic_img.size
        
        # Create a properly sized crop with padding if needed
        wms_crop = Image.new("RGB", (crop_size, crop_size), (20, 20, 20))
        
        # Calculate source and destination regions
        src_l = max(0, crop_l)
        src_t = max(0, crop_t)
        src_r = min(ms_w, crop_r)
        src_b = min(ms_h, crop_b)
        
        dst_l = max(0, -crop_l)
        dst_t = max(0, -crop_t)
        
        if src_r > src_l and src_b > src_t:
            region = mosaic_img.crop((src_l, src_t, src_r, src_b))
            wms_crop.paste(region, (dst_l, dst_t))
        
        # Rotate WMS using the SAME function and target_size as ortho
        wms_rotated = _rotate_image(wms_crop, heading if heading else 0, (width, height))
        base_frame = wms_rotated.convert("RGBA")
        
    except Exception as e:
        print(f"WMS Fail: {e}")
        base_frame = Image.new("RGBA", (width, height), (20, 20, 20, 255))

    clipped = clip_vectors(_GLOBAL_VECTORS, bbox)
    
    # Render ortho layer (uses same geometry internally)
    frame = render_frame(
        _GLOBAL_DATASET,
        clipped,
        center_e,
        center_n,
        heading,
        width,
        height,
        map_half_width_m,
        arrow_size_px,
        cone_angle_deg,
        cone_length_px,
        cone_opacity,
        icon_circle_opacity,
        icon_circle_size_px,
        show_compass,
        compass_size_px,
    )
    
    if frame.mode != "RGBA":
        frame = frame.convert("RGBA")
    
    if base_frame.size != (width, height):
        base_frame = base_frame.resize((width, height), Image.LANCZOS)

    # Composite: ortho over WMS
    final_image = Image.alpha_composite(base_frame, frame)
    final_image.save(frame_path, "PNG")
    return 1

