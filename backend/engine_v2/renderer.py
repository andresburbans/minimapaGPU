from typing import Any, Dict, Iterable, List, Tuple

try:
    from render_gpu_backup_v2 import (
        HAS_GPU as _HAS_GPU,
        cleanup_gpu as _cleanup_gpu,
        init_gpu as _init_gpu,
        preload_track_gpu as _preload_track_gpu,
        render_frame_gpu as _render_frame_gpu,
    )

    ENGINE_V2_AVAILABLE = True
except ImportError:
    _HAS_GPU = False
    ENGINE_V2_AVAILABLE = False

    def _init_gpu() -> Dict[str, Any]:
        return {"available": False, "backend": "none"}

    def _preload_track_gpu(config: Any, jobs: List[Tuple], progress_callback=None) -> None:
        return None

    def _render_frame_gpu(*args, **kwargs):
        raise NotImplementedError("Engine v2 backend not available")

    def _cleanup_gpu() -> None:
        return None


def init_engine_v2(prefer_gpu: bool = True) -> Dict[str, Any]:
    if not ENGINE_V2_AVAILABLE:
        return {"available": False, "backend": "none", "using_cuda": False}

    info = _init_gpu()
    available = bool(_HAS_GPU and info.get("available", False))
    return {
        "available": available,
        "backend": "render_gpu_backup_v2" if available else info.get("backend", "cpu"),
        "device": info.get("device"),
        "count": info.get("count"),
        "memory_free": info.get("memory_free"),
        "memory_total": info.get("memory_total"),
        "using_cuda": available and bool(prefer_gpu),
    }


def preload_track_v2(config: Any, jobs: List[Tuple], progress_callback=None) -> None:
    if not ENGINE_V2_AVAILABLE:
        return
    _preload_track_gpu(config, jobs, progress_callback=progress_callback)


def render_frame_v2(
    dataset,
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
):
    return _render_frame_gpu(
        dataset=dataset,
        vectors=vectors,
        center_e=center_e,
        center_n=center_n,
        heading=heading,
        width=width,
        height=height,
        map_half_width_m=map_half_width_m,
        arrow_size_px=arrow_size_px,
        cone_angle_deg=cone_angle_deg,
        cone_length_px=cone_length_px,
        cone_opacity=cone_opacity,
        icon_circle_opacity=icon_circle_opacity,
        icon_circle_size_px=icon_circle_size_px,
        show_compass=show_compass,
        compass_size_px=compass_size_px,
        wms_source=wms_source,
    )


def cleanup_v2() -> None:
    _cleanup_gpu()
