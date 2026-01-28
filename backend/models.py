from pydantic import BaseModel, Field
from typing import List, Optional


class VectorLayer(BaseModel):
    path: str
    color: str = "#29b3ff"
    width: int = 3
    pattern: str = "solid"


class TrackPoint(BaseModel):
    t: float
    x: float
    y: float


class TrackRequest(BaseModel):
    video_path: str
    points: List[TrackPoint]
    output_name: str = "track.json"


class OverlayRequest(BaseModel):
    video_path: str
    track_path: str
    output_name: str = "overlay.mp4"
    output_dir: Optional[str] = None
    line_color: str = "#23c4ff"
    line_width: int = 3
    point_color: str = "#ffffff"
    show_points: bool = False
    snap_to_edges: bool = True
    snap_radius: Optional[int] = None
    smooth_alpha: float = 0.65
    max_jump: Optional[float] = None


class RenderConfig(BaseModel):
    ortho_path: str
    csv_path: str
    vectors_paths: List[str] = Field(default_factory=list)
    vector_layers: List[VectorLayer] = Field(default_factory=list)
    curves_path: Optional[str] = None
    fps: int = 30
    duration_sec: float
    width: int = 2048
    height: int = 2048
    map_half_width_m: float = 150.0
    arrow_size_px: int = 400
    cone_angle_deg: float = 60.0
    cone_length_px: int = 220
    cone_opacity: float = 0.18
    icon_circle_opacity: float = 0.35
    icon_circle_size_px: int = 120
    show_compass: bool = True
    compass_size_px: int = 40
    line_color: str = "#29b3ff"
    line_width: int = 3
    boundary_color: str = "#ffffff"
    boundary_width: int = 4
    point_color: str = "#ff9f1c"
    background_opacity: float = 1.0
    use_gpu: bool = False
    workers: int = 0
    output_name: str = "minimapa.mp4"


class PreviewRequest(BaseModel):
    config: RenderConfig
    time_sec: float
