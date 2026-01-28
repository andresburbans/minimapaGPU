import json
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np


def track_points(
    video_path: str,
    points: List[dict],
    output_path: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    points_sorted = sorted(list(enumerate(points)), key=lambda item: item[1]["t"])
    total_points = len(points_sorted)

    active_ids: List[int] = []
    positions: List[Optional[np.ndarray]] = [None] * total_points
    next_point_idx = 0

    frames = []
    prev_gray = None
    frame_idx = 0

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t = frame_idx / fps

        if prev_gray is not None and active_ids:
            p0 = np.array([positions[idx] for idx in active_ids], dtype=np.float32).reshape(-1, 1, 2)
            p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            new_active: List[int] = []
            for i, pid in enumerate(active_ids):
                if st[i][0] == 1:
                    positions[pid] = p1[i][0]
                    new_active.append(pid)
                else:
                    positions[pid] = None
            active_ids = new_active

        while next_point_idx < total_points and points_sorted[next_point_idx][1]["t"] <= t:
            pid, point = points_sorted[next_point_idx]
            positions[pid] = np.array([point["x"] * width, point["y"] * height], dtype=np.float32)
            active_ids.append(pid)
            next_point_idx += 1

        frame_points = []
        for pos in positions:
            if pos is None:
                frame_points.append(None)
            else:
                frame_points.append([float(pos[0] / width), float(pos[1] / height)])

        frames.append({"t": t, "points": frame_points})
        prev_gray = gray
        frame_idx += 1

        if progress_cb and (frame_idx % 10 == 0 or frame_idx == total_frames):
            progress_cb(frame_idx, total_frames if total_frames > 0 else frame_idx)

    cap.release()

    data = {
        "fps": fps,
        "width": width,
        "height": height,
        "points": points,
        "frames": frames,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return output_path


def render_overlay(
    video_path: str,
    track_path: str,
    output_path: str,
    line_color: str,
    line_width: int,
    point_color: str,
    show_points: bool = False,
    snap_to_edges: bool = True,
    snap_radius: Optional[int] = None,
    smooth_alpha: float = 0.65,
    max_jump: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    import subprocess
    import tempfile
    
    with open(track_path, "r", encoding="utf-8") as f:
        track = json.load(f)

    frames_data = track.get("frames", [])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or track.get("fps", 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(frames_data)

    # Usar carpeta temporal para frames procesados
    temp_dir = Path(tempfile.mkdtemp(prefix="overlay_"))
    
    line_bgr = _hex_to_bgr(line_color)
    point_bgr = _hex_to_bgr(point_color)
    alpha = max(0.0, min(0.95, smooth_alpha))
    if snap_radius is None:
        snap_radius = max(6, int(min(width, height) * 0.007))
    if max_jump is None:
        max_jump = max(20.0, min(width, height) * 0.02)

    smooth_positions = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frames_data):
            break
        frame_info = frames_data[frame_idx]
        points = frame_info.get("points", [])
        if smooth_positions is None:
            smooth_positions = [None] * len(points)

        edges = None
        if snap_to_edges:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

        poly_points = []
        for idx, pt in enumerate(points):
            if pt is None:
                smooth_positions[idx] = None
                continue
            x = int(pt[0] * width)
            y = int(pt[1] * height)
            if edges is not None:
                x, y = _snap_point_to_edges(edges, x, y, snap_radius)
            prev = smooth_positions[idx]
            if prev is not None:
                dist = float(np.hypot(x - prev[0], y - prev[1]))
                if dist > max_jump * 2:
                    smooth_positions[idx] = None
                    continue
                if dist > max_jump:
                    x, y = int(prev[0]), int(prev[1])
                x = int(prev[0] * alpha + x * (1 - alpha))
                y = int(prev[1] * alpha + y * (1 - alpha))
            smooth_positions[idx] = (x, y)
            poly_points.append((x, y))
            if show_points:
                cv2.circle(frame, (x, y), max(2, line_width + 1), point_bgr, -1)

        if len(poly_points) > 1:
            for i in range(1, len(poly_points)):
                cv2.line(frame, poly_points[i - 1], poly_points[i], line_bgr, line_width)

        # Guardar frame procesado como PNG
        frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

        if progress_cb and (frame_idx % 10 == 0 or frame_idx == total_frames):
            progress_cb(frame_idx, total_frames if total_frames > 0 else frame_idx)

    cap.release()
    
    # Codificar con FFmpeg usando GPU si está disponible
    _encode_overlay_video(temp_dir, Path(output_path), fps)
    
    # Limpiar frames temporales
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path


def _encode_overlay_video(input_dir: Path, output_path: Path, fps: float):
    """Codifica frames a video usando NVENC si está disponible"""
    import subprocess
    import sys
    
    # Importar gpu_utils desde el directorio backend
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import gpu_utils
        gpu_info = gpu_utils.detect_cuda_gpu()
    except ImportError:
        gpu_info = {"nvenc_available": False}
    
    # Configurar codec y parámetros
    if gpu_info.get("nvenc_available", False):
        codec = "h264_nvenc"
        extra_args = [
            "-preset", "p5",     # Preset balanceado (p1=rápido, p7=lento/mejor)
            "-rc:v", "vbr",      # Variable bitrate
            "-cq", "20",         # Calidad constante (0-51, 20 = muy alta calidad)
            "-b:v", "0",         # Dejar que CQ controle la calidad
            "-gpu", "0",         # Usar primera GPU (ya forzada por gpu_utils)
        ]
        print(f"[GPU] Codificando overlay con NVENC (GPU acelerada)")
    else:
        codec = "libx264"
        extra_args = [
            "-preset", "medium", # Preset balanceado
            "-crf", "20",        # Calidad constante (0-51, 20 = muy alta calidad)
        ]
        print(f"[CPU] Codificando overlay con libx264 (software)")

    command = [
        "ffmpeg",
        "-y",                              # Sobrescribir archivo de salida
        "-framerate", str(fps),            # FPS del video de entrada
        "-i", str(input_dir / "frame_%06d.png"),  # Patrón de frames PNG
        "-c:v", codec,                     # Codec de video
        *extra_args,                       # Argumentos específicos del codec
        "-pix_fmt", "yuv420p",             # Formato compatible universal
        "-movflags", "+faststart",         # Optimizar para streaming web
        "-vsync", "cfr",                   # Constant frame rate (evita lag)
        str(output_path),
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[ENCODE] Video overlay generado: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Fallo la codificación del video")
        print(f"Comando: {' '.join(command)}")
        if e.stderr:
            print(f"FFmpeg stderr:\n{e.stderr}")
        raise RuntimeError(f"Fallo la codificación del video: {e}")


def _snap_point_to_edges(edges: np.ndarray, x: int, y: int, radius: int) -> tuple:
    h, w = edges.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(h - 1, y + radius)
    roi = edges[y0 : y1 + 1, x0 : x1 + 1]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return x, y
    dx = xs - (x - x0)
    dy = ys - (y - y0)
    idx = int(np.argmin(dx * dx + dy * dy))
    return int(x0 + xs[idx]), int(y0 + ys[idx])


def _hex_to_bgr(value: str) -> tuple:
    hex_value = value.strip().lstrip("#")
    if len(hex_value) != 6:
        return (255, 255, 255)
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    return (b, g, r)
