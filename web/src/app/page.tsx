"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";

// ==================== TYPES ====================
type UploadRef = {
  path: string;
  filename: string;
};

type RoutePoint = {
  id: string;
  E_ini: number;
  N_ini: number;
  E_fin: number;
  N_fin: number;
  t_start: number;
  t_end: number;
  heading: number;
};

type RenderConfig = {
  ortho_path: string;
  csv_path: string;
  vectors_paths: string[];
  vector_layers: { path: string; color: string; width: number; pattern: string }[];
  curves_path?: string | null;
  fps: number;
  duration_sec: number;
  width: number;
  height: number;
  map_half_width_m: number;
  arrow_size_px: number;
  cone_angle_deg: number;
  cone_length_px: number;
  cone_opacity: number;
  icon_circle_opacity: number;
  icon_circle_size_px: number;
  show_compass: boolean;
  compass_size_px: number;
  line_color: string;
  line_width: number;
  boundary_color: string;
  boundary_width: number;
  point_color: string;
  output_name: string;
  use_gpu: boolean;
  workers: number;
};

type VectorGeometry = {
  paths: [number, number][][];
  points: [number, number][];
};

type VectorInput = {
  id: string;
  label: string;
  file: File | null;
  color: string;
  width: number;
  pattern: string;
  visible: boolean;
  ref?: UploadRef | null;
  geometry?: VectorGeometry | null;
  uploading?: boolean;
  error?: string | null;
};

type PathPoint = {
  t: number;
  x: number;
  y: number;
};

type TrackFrame = {
  t: number;
  points: (number[] | null)[];
};

type TrackData = {
  fps: number;
  width: number;
  height: number;
  points: PathPoint[];
  frames: TrackFrame[];
};

type JobStatus = {
  status: string;
  progress?: number;
  total?: number;
  message?: string;
  log?: string[];
  output_path?: string;
  track_path?: string;
  error?: string;
};

type OrthoMetadata = {
  width: number;
  height: number;
  bounds: [number, number, number, number]; // [minE, minN, maxE, maxN]
  transform: number[];
};

type PickerMode = "start" | "end" | null;

type SystemStats = {
  cpu_usage: number;
  ram_usage: number;
  gpu_available: boolean;
  gpu_usage: number;
  gpu_temp: number;
  gpu_name: string;
  nvenc_available: boolean;
};

type FileSystemFileHandle = {
  name: string;
  createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }>;
};

type TimeDraft = {
  t_start: string;
  t_end: string;
};

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type ShowSaveFilePicker = (options?: SaveFilePickerOptions) => Promise<FileSystemFileHandle>;

// ==================== CONSTANTS ====================
const API = "http://127.0.0.1:8000";
const DEFAULT_LAYER_COLORS = ["#EF0B85", "#616161", "#47AFFF", "#ffd166", "#d36cff", "#00d1b2"];
const DEFAULT_LINE_COLOR = "#29b3ff";
const DEFAULT_LINE_WIDTH = 3;
const DEFAULT_BOUNDARY_COLOR = "#ffffff";
const DEFAULT_BOUNDARY_WIDTH = 4;
const DEFAULT_POINT_COLOR = "#ff9f1c";
const DEFAULT_ICON_SIZE = 400;
const DEFAULT_ICON_CIRCLE_OPACITY = 0.35;
const DEFAULT_ICON_CIRCLE_SIZE = 120;
const LINE_STYLES = [
  { value: "solid", label: "Sólido" },
  { value: "dashed", label: "Guiones" },
  { value: "dotted", label: "Punteado" },
  { value: "dashdot", label: "Guión punto" },
  { value: "road-arrows", label: "Vía con flechas" },
  { value: "tactical", label: "Lindero táctico" },
];
const VECTOR_PRESETS = [
  { id: "lindero-general", label: "Lindero general", color: "#EF0B85", width: 3, pattern: "solid", visible: true },
  { id: "vias", label: "Vias", color: "#616161", width: 3, pattern: "solid", visible: true },
  { id: "quebradas", label: "Quebradas", color: "#47AFFF", width: 3, pattern: "solid", visible: true },
];

// ==================== UTILITY FUNCTIONS ====================
function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

function computeHeading(E1: number, N1: number, E2: number, N2: number): number {
  const dx = E2 - E1;
  const dy = N2 - N1;
  const angleRad = Math.atan2(dx, dy);
  return (((angleRad * 180) / Math.PI) + 360) % 360;
}

function formatTime(seconds: number): string {
  const clamped = Math.max(0, Math.min(600, seconds));
  const mins = Math.floor(clamped / 60);
  const secs = Math.floor(clamped % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

function getPointLabel(index: number, type: "start" | "end"): string {
  const letter = type === "start" ? "A" : "B";
  return `${letter}${index + 1}`;
}

function clampTime(seconds: number): number {
  return Math.max(0, Math.min(600, seconds));
}

function parseTimeInput(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!/^\d{1,2}(:\d{1,2})?$/.test(trimmed)) return null;
  const parts = trimmed.split(":");
  let mins = 0;
  let secs = 0;
  if (parts.length === 1) {
    secs = Number(parts[0]);
  } else {
    mins = Number(parts[0]);
    secs = Number(parts[1]);
  }
  if (Number.isNaN(mins) || Number.isNaN(secs) || secs >= 60) return null;
  return clampTime(mins * 60 + secs);
}

function toCoordPair(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return [x, y];
}

function collectGeometry(geom: unknown, paths: [number, number][][], points: [number, number][]): void {
  if (!geom || typeof geom !== "object") return;
  const g = geom as { type?: string; coordinates?: unknown; geometries?: unknown[] };
  switch (g.type) {
    case "LineString": {
      const coords = (Array.isArray(g.coordinates) ? g.coordinates : []).map(toCoordPair).filter(Boolean) as [number, number][];
      if (coords.length > 1) paths.push(coords);
      break;
    }
    case "MultiLineString": {
      const lines = Array.isArray(g.coordinates) ? g.coordinates : [];
      for (const line of lines) {
        const coords = (Array.isArray(line) ? line : []).map(toCoordPair).filter(Boolean) as [number, number][];
        if (coords.length > 1) paths.push(coords);
      }
      break;
    }
    case "Polygon": {
      const rings = Array.isArray(g.coordinates) ? g.coordinates : [];
      for (const ring of rings) {
        const coords = (Array.isArray(ring) ? ring : []).map(toCoordPair).filter(Boolean) as [number, number][];
        if (coords.length > 1) paths.push(coords);
      }
      break;
    }
    case "MultiPolygon": {
      const polys = Array.isArray(g.coordinates) ? g.coordinates : [];
      for (const poly of polys) {
        const rings = Array.isArray(poly) ? poly : [];
        for (const ring of rings) {
          const coords = (Array.isArray(ring) ? ring : []).map(toCoordPair).filter(Boolean) as [number, number][];
          if (coords.length > 1) paths.push(coords);
        }
      }
      break;
    }
    case "Point": {
      const coord = toCoordPair(g.coordinates);
      if (coord) points.push(coord);
      break;
    }
    case "MultiPoint": {
      const pts = Array.isArray(g.coordinates) ? g.coordinates : [];
      for (const pt of pts) {
        const coord = toCoordPair(pt);
        if (coord) points.push(coord);
      }
      break;
    }
    case "GeometryCollection": {
      const geoms = Array.isArray(g.geometries) ? g.geometries : [];
      for (const subGeom of geoms) {
        collectGeometry(subGeom, paths, points);
      }
      break;
    }
  }
}

function parseGeoJSON(data: unknown): VectorGeometry {
  const paths: [number, number][][] = [];
  const points: [number, number][] = [];
  if (!data || typeof data !== "object") return { paths, points };
  const root = data as { type?: string; features?: unknown[]; geometry?: unknown };
  if (root.type === "FeatureCollection") {
    const features = Array.isArray(root.features) ? root.features : [];
    for (const feature of features) {
      const featGeom = feature && typeof feature === "object" ? (feature as { geometry?: unknown }).geometry : undefined;
      collectGeometry(featGeom, paths, points);
    }
  } else if (root.type === "Feature") {
    collectGeometry(root.geometry, paths, points);
  } else {
    collectGeometry(root, paths, points);
  }
  return { paths, points };
}

function dashArrayForPattern(pattern: string, scale: number): string | undefined {
  const s = Math.max(scale, 0.1);
  switch ((pattern || "solid").toLowerCase()) {
    case "dashed":
      return `${12 * s} ${8 * s}`;
    case "dotted":
      return `${2 * s} ${6 * s}`;
    case "dashdot":
      return `${12 * s} ${6 * s} ${2 * s} ${6 * s}`;
    case "tactical":
      return `${16 * s} ${6 * s} ${2 * s} ${6 * s}`;
    default:
      return undefined;
  }
}

function buildPathD(points: [number, number][], toPixel: (x: number, y: number) => { x: number; y: number }): string {
  if (!points.length) return "";
  return points
    .map((pt, idx) => {
      const pos = toPixel(pt[0], pt[1]);
      return `${idx === 0 ? "M" : "L"}${pos.x} ${pos.y}`;
    })
    .join(" ");
}

function createDefaultVectorInputs(): VectorInput[] {
  return VECTOR_PRESETS.map((layer) => ({
    ...layer,
    file: null,
    ref: null,
    geometry: null,
    uploading: false,
    error: null,
    visible: true,
  }));
}

// ==================== MAIN COMPONENT ====================
export default function Home() {
  // ========== APP STATE ==========
  const [appPhase, setAppPhase] = useState<"upload" | "editor">("upload");
  const [mode, setMode] = useState<"minimap" | "path">("minimap");
  const [backendOnline, setBackendOnline] = useState(false);

  // ========== UPLOAD PHASE (MINIMAP) ==========
  const [orthoFile, setOrthoFile] = useState<File | null>(null);
  const [vectorInputs, setVectorInputs] = useState<VectorInput[]>(() => createDefaultVectorInputs());
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadLogs, setUploadLogs] = useState<string[]>([]);

  // ========== UPLOADED REFS (MINIMAP) ==========
  const [orthoRef, setOrthoRef] = useState<UploadRef | null>(null);
  const [orthoMetadata, setOrthoMetadata] = useState<OrthoMetadata | null>(null);
  const [orthoImageUrl, setOrthoImageUrl] = useState<string | null>(null);
  const [wmsImageUrl, setWmsImageUrl] = useState<string | null>(null);
  const [baseLayer, setBaseLayer] = useState<"ortho" | "wms">("wms");

  // ========== ROUTE POINTS (MINIMAP) ==========
  const [routePoints, setRoutePoints] = useState<RoutePoint[]>([]);
  const [pickerMode, setPickerMode] = useState<PickerMode>(null);
  const [editingPointId, setEditingPointId] = useState<string | null>(null);
  const [autoAdvancePicker, setAutoAdvancePicker] = useState(false);
  const [timeDrafts, setTimeDrafts] = useState<Record<string, TimeDraft>>({});

  // ========== MAP VIEWER STATE (MINIMAP) ==========
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapImageRef = useRef<HTMLImageElement>(null);
  const [mapZoom, setMapZoom] = useState(1);
  const [mapOffset, setMapOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  // mapSize = dimensiones renderizadas de la imagen
  const [mapContainerSize, setMapContainerSize] = useState({ width: 0, height: 0 });
  const [mapSize, setMapSize] = useState({ width: 0, height: 0 });
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });

  // ========== PREVIEW STATE (MINIMAP) ==========
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewTime, setPreviewTime] = useState(0);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // ========== RENDER SETTINGS (MINIMAP) ==========
  const [fps, setFps] = useState(30);
  const [width, setWidth] = useState(640);
  const [height, setHeight] = useState(360);
  const [mapHalfWidth, setMapHalfWidth] = useState(150);
  const [arrowSize, setArrowSize] = useState(DEFAULT_ICON_SIZE);
  const [coneAngle, setConeAngle] = useState(60);
  const [coneLength, setConeLength] = useState(220);
  const [coneOpacity, setConeOpacity] = useState(0.18);
  const [iconCircleOpacity, setIconCircleOpacity] = useState(DEFAULT_ICON_CIRCLE_OPACITY);
  const [iconCircleSize, setIconCircleSize] = useState(DEFAULT_ICON_CIRCLE_SIZE);
  const [mapZoomFactor, setMapZoomFactor] = useState(1);
  const [showCompass, setShowCompass] = useState(true);
  const [compassSize, setCompassSize] = useState(40);
  const [outputName, setOutputName] = useState("minimapa.mp4");
  const [useGpu, setUseGpu] = useState(false);

  // ========== PATH MODE STATE ==========
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoRef, setVideoRef] = useState<UploadRef | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [annotating, setAnnotating] = useState(false);
  const [pathError, setPathError] = useState<string | null>(null);
  const [pathPoints, setPathPoints] = useState<PathPoint[]>([]);
  const [pathLineColor, setPathLineColor] = useState("#ffffff");
  const [pathLineWidth, setPathLineWidth] = useState(3);
  const [saveHandle, setSaveHandle] = useState<FileSystemFileHandle | null>(null);
  const [saveFileName, setSaveFileName] = useState<string | null>(null);
  const [videoAspect, setVideoAspect] = useState(16 / 9);
  const [playerWidth, setPlayerWidth] = useState<number | null>(null);
  const [viewportHeight, setViewportHeight] = useState(0);
  const playerWrapRef = useRef<HTMLDivElement>(null);
  const [pathJobId, setPathJobId] = useState<string | null>(null);
  const [pathJobStatus, setPathJobStatus] = useState<JobStatus | null>(null);
  const [trackData, setTrackData] = useState<TrackData | null>(null);
  const [trackPath, setTrackPath] = useState<string | null>(null);
  const [overlayJobId, setOverlayJobId] = useState<string | null>(null);
  const [overlayJobStatus, setOverlayJobStatus] = useState<JobStatus | null>(null);
  const videoElement = useRef<HTMLVideoElement>(null);
  const overlayCanvas = useRef<HTMLCanvasElement>(null);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // ========== EXPORT STATE ==========
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [exportPath, setExportPath] = useState<string | null>(null);
  const [showVideoModal, setShowVideoModal] = useState(false);
  const [minimapExportPath, setMinimapExportPath] = useState<string | null>(null);
  const [minimapSaveHandle, setMinimapSaveHandle] = useState<FileSystemFileHandle | null>(null);
  const [minimapSaveName, setMinimapSaveName] = useState<string | null>(null);
  const [minimapSaveError, setMinimapSaveError] = useState<string | null>(null);
  const [minimapSaving, setMinimapSaving] = useState(false);
  const [minimapSavedPath, setMinimapSavedPath] = useState<string | null>(null);

  // ========== COMPUTED VALUES ==========
  const duration = useMemo(() => {
    if (mode === "minimap") {
      if (routePoints.length === 0) return 0;
      return Math.max(...routePoints.map((p) => p.t_end));
    }
    return 0; // Duration is handled by video in path mode
  }, [routePoints, mode]);

  const activeVectorLayers = useMemo(() => {
    return [...vectorInputs]
      .reverse()
      .filter((layer) => layer.ref && layer.visible)
      .map((layer) => ({
        path: layer.ref!.path,
        color: layer.color,
        width: layer.width,
        pattern: layer.pattern,
      }));
  }, [vectorInputs]);

  const canPickMinimapSave = typeof window !== "undefined" && "showSaveFilePicker" in window;

  const pickerLabel = useMemo(() => {
    if (!pickerMode) return "";
    const index = routePoints.findIndex((p) => p.id === editingPointId);
    if (index >= 0) return getPointLabel(index, pickerMode === "start" ? "start" : "end");
    return pickerMode === "start" ? "A" : "B";
  }, [pickerMode, editingPointId, routePoints]);

  const renderBusy = jobStatus?.status === "queued" || jobStatus?.status === "rendering";
  const isPicking = pickerMode !== null || editingPointId !== null;
  const vectorScale = 1 / Math.max(0.2, mapZoom);

  useEffect(() => {
    setTimeDrafts((prev) => {
      const next: Record<string, TimeDraft> = {};
      for (const p of routePoints) {
        next[p.id] = {
          t_start: prev[p.id]?.t_start ?? formatTime(p.t_start),
          t_end: prev[p.id]?.t_end ?? formatTime(p.t_end),
        };
      }
      return next;
    });
  }, [routePoints]);

  const csvContent = useMemo(() => {
    const header = "t_start,t_end,E_ini,N_ini,E_fin,N_fin,heading\n";
    const rows = routePoints
      .map((p) => `${p.t_start},${p.t_end},${p.E_ini},${p.N_ini},${p.E_fin},${p.N_fin},${p.heading.toFixed(2)}`)
      .join("\n");
    return header + rows;
  }, [routePoints]);

  // ========== BACKEND HEALTH CHECK ==========
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      try {
        const res = await fetch(`${API}/health`);
        if (alive) setBackendOnline(res.ok);
      } catch {
        if (alive) setBackendOnline(false);
      }
    };
    ping();
    const timer = setInterval(ping, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  // ========== VIEWPORT AND VIDEO EFFECTS ==========
  useEffect(() => {
    const update = () => setViewportHeight(window.innerHeight);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  useEffect(() => {
    if (!videoFile) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    if (!videoFile) return;
    setAnnotating(false);
    setPathPoints([]);
    setTrackData(null);
    setTrackPath(null);
    setPathJobId(null);
    setPathJobStatus(null);
    setOverlayJobId(null);
    setOverlayJobStatus(null);
    setExportPath(null);
  }, [videoFile]);

  // ========== MAP CONTAINER SIZE (MINIMAP) ==========
  useEffect(() => {
    if (!mapContainerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setMapContainerSize({ width, height });
    });
    observer.observe(mapContainerRef.current);
    return () => observer.disconnect();
  }, [appPhase, mode]);

  useEffect(() => {
    const el = mapContainerRef.current;
    if (!el || mode !== "minimap") return;
    const blockScroll = (event: WheelEvent) => {
      event.preventDefault();
    };
    el.addEventListener("wheel", blockScroll, { passive: false });
    return () => {
      el.removeEventListener("wheel", blockScroll);
    };
  }, [mode]);

  useEffect(() => {
    if (!mapContainerSize.width || !mapContainerSize.height) return;
    if (!naturalSize.width || !naturalSize.height) return;
    const aspect = naturalSize.width / naturalSize.height;

    // COVER LOGIC: Aprovechar la totalidad del recuadro
    let renderWidth = mapContainerSize.width;
    let renderHeight = mapContainerSize.height;

    if (mapContainerSize.width / mapContainerSize.height > aspect) {
      // Contenedor "más apaisado" que la imagen -> Ajustar al ancho (sobra alto)
      renderWidth = mapContainerSize.width;
      renderHeight = renderWidth / aspect;
    } else {
      // Contenedor "más vertical" que la imagen -> Ajustar al alto (sobra ancho)
      renderHeight = mapContainerSize.height;
      renderWidth = renderHeight * aspect;
    }

    setMapSize({ width: renderWidth, height: renderHeight });
    setMapOffset({
      x: (mapContainerSize.width - renderWidth) / 2,
      y: (mapContainerSize.height - renderHeight) / 2,
    });
  }, [mapContainerSize, naturalSize]);

  // ========== JOB STATUS POLLING (MINIMAP RENDER) ==========
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    const poll = async () => {
      try {
        const res = await fetch(`${API}/status?job_id=${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        setJobStatus(data);
        if (data.status === "finished" && data.output_path) {
          setMinimapExportPath(data.output_path);
          setShowVideoModal(true);
          return;
        }
        if (data.status === "error" || data.status === "not_found") return;
        setTimeout(poll, 1000);
      } catch {
        setTimeout(poll, 1500);
      }
    };
    poll();
    return () => { alive = false; };
  }, [jobId]);

  // ========== PATH JOB POLLING ==========
  useEffect(() => {
    if (!pathJobId) return;
    let alive = true;
    const poll = async () => {
      try {
        const res = await fetch(`${API}/status?job_id=${pathJobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        setPathJobStatus(data);
        if (data.status === "finished" && data.track_path) {
          setTrackPath(data.track_path);
          const trackRes = await fetch(`${API}/file?path=${encodeURIComponent(data.track_path)}`);
          if (trackRes.ok) {
            const trackJson = await trackRes.json();
            if (alive) setTrackData(trackJson);
          }
          return;
        }
        if (data.status === "error" || data.status === "not_found") return;
        setTimeout(poll, 1000);
      } catch {
        setTimeout(poll, 1500);
      }
    };
    poll();
    return () => { alive = false; };
  }, [pathJobId]);

  // ========== OVERLAY JOB POLLING ==========
  useEffect(() => {
    if (!overlayJobId) return;
    let alive = true;
    const poll = async () => {
      try {
        const res = await fetch(`${API}/status?job_id=${overlayJobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        setOverlayJobStatus(data);
        if (data.status === "finished" && data.output_path) {
          setExportPath(data.output_path);
          return;
        }
        if (data.status === "error" || data.status === "not_found") return;
        setTimeout(poll, 1000);
      } catch {
        setTimeout(poll, 1500);
      }
    };
    poll();
    return () => { alive = false; };
  }, [overlayJobId]);

  // ========== SYSTEM STATS POLLING ==========
  useEffect(() => {
    if (!backendOnline) {
      setSystemStats(null);
      return;
    }
    let alive = true;
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API}/stats`);
        if (res.ok) {
          const data = await res.json();
          if (alive) setSystemStats(data);
        }
      } catch {
        if (alive) setSystemStats(null);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, [backendOnline]);

  // ========== AUTO-SCROLL LOGS ==========
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [jobStatus?.log, pathJobStatus?.log, overlayJobStatus?.log]);

  // ========== CANVAS DRAWING LOOP (PATH MODE) ==========
  useEffect(() => {
    if (mode !== "path") return;
    let rafId = 0;
    const draw = () => {
      const video = videoElement.current;
      const canvas = overlayCanvas.current;
      if (video && canvas) {
        const width = video.clientWidth;
        const height = video.clientHeight;
        if (width > 0 && height > 0) {
          if (canvas.width != width) canvas.width = width;
          if (canvas.height != height) canvas.height = height;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, width, height);
            const current = video.currentTime;

            // Draw track if available
            if (trackData && trackData.frames && trackData.frames.length) {
              const idx = Math.min(trackData.frames.length - 1, Math.floor(current * trackData.fps));
              const frame = trackData.frames[idx];
              if (frame) {
                const pts = frame.points || [];
                let started = false;
                ctx.strokeStyle = pathLineColor;
                ctx.lineWidth = pathLineWidth;
                ctx.lineJoin = "round";
                pts.forEach((pt) => {
                  if (!pt) {
                    started = false;
                    return;
                  }
                  const x = pt[0] * width;
                  const y = pt[1] * height;
                  if (!started) {
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    started = true;
                  } else {
                    ctx.lineTo(x, y);
                  }
                });
                if (started) ctx.stroke();
              }
            }

            // Draw point indicators
            if (pathPoints.length && (annotating || !trackData)) {
              const windowSec = 2;
              pathPoints.forEach((pt) => {
                const age = current - pt.t;
                if (age < 0 || age > windowSec) return;
                const alpha = 1 - age / windowSec;
                const x = pt.x * width;
                const y = pt.y * height;
                ctx.strokeStyle = pathLineColor;
                ctx.lineWidth = Math.max(2, pathLineWidth - 1);
                ctx.fillStyle = `rgba(255,255,255,${alpha})`;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
              });
            }
          }
        }
      }
      rafId = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(rafId);
  }, [mode, pathPoints, trackData, pathLineColor, pathLineWidth, annotating]);

  // ========== PLAYER RESIZE LOGIC ==========
  useEffect(() => {
    if (!playerWrapRef.current) return;
    const update = () => {
      if (!playerWrapRef.current) return;
      const wrapWidth = playerWrapRef.current.clientWidth;
      if (wrapWidth <= 0) return;
      const maxHeight = (viewportHeight || window.innerHeight) * 0.68;
      const width = Math.min(wrapWidth, maxHeight * videoAspect);
      setPlayerWidth(width);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(playerWrapRef.current);
    return () => observer.disconnect();
  }, [videoAspect, viewportHeight, mode]);

  // ========== UPLOAD FUNCTIONS ==========
  const uploadFile = async (file: File): Promise<UploadRef> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error("No se pudo subir el archivo");
    return res.json();
  };

  const uploadFileWithProgress = (file: File, onProgress: (loaded: number) => void): Promise<UploadRef> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/upload`);
      xhr.responseType = "json";
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(event.loaded);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.response);
        } else {
          reject(new Error("No se pudo subir el archivo"));
        }
      };
      xhr.onerror = () => reject(new Error("No se pudo subir el archivo"));
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    });
  };

  const handleVectorFileChange = async (index: number, file: File | null) => {
    if (!file) {
      updateVectorInput(index, { file: null, ref: null, geometry: null, uploading: false, error: null });
      return;
    }

    updateVectorInput(index, { file, ref: null, geometry: null, uploading: true, error: null });

    let geometry: VectorGeometry | null = null;
    try {
      const text = await file.text();
      geometry = parseGeoJSON(JSON.parse(text));
      setVectorInputs((prev) =>
        prev.map((layer, i) => (i === index && layer.file === file ? { ...layer, geometry } : layer))
      );
    } catch {
      updateVectorInput(index, { uploading: false, error: "No se pudo leer el GeoJSON." });
      return;
    }

    try {
      const ref = await uploadFile(file);
      setVectorInputs((prev) =>
        prev.map((layer, i) =>
          i === index && layer.file === file ? { ...layer, ref, uploading: false, error: null } : layer
        )
      );
    } catch {
      setVectorInputs((prev) =>
        prev.map((layer, i) =>
          i === index && layer.file === file ? { ...layer, uploading: false, error: "No se pudo subir el vector." } : layer
        )
      );
    }
  };

  const handleUploadAndStart = async () => {
    if (!orthoFile) {
      setUploadError("Debes seleccionar un ortomosaico GeoTIFF");
      return;
    }
    setUploading(true);
    setUploadError(null);
    setUploadProgress(0);
    setUploadLogs(["Iniciando carga..."]);

    try {
      const totalBytes = orthoFile.size;
      let uploadedBytes = 0;
      const updateProgress = (loaded: number) => {
        if (!totalBytes) return;
        const pct = Math.min(98, Math.round(((uploadedBytes + loaded) / totalBytes) * 100));
        setUploadProgress((prev) => Math.max(prev, pct));
      };
      const log = (msg: string) => setUploadLogs((prev) => [...prev.slice(-6), msg]);

      log("Subiendo ortomosaico...");
      const ortho = await uploadFileWithProgress(orthoFile, updateProgress);
      uploadedBytes += orthoFile.size;
      setUploadProgress((prev) => Math.max(prev, Math.round((uploadedBytes / totalBytes) * 100)));
      setOrthoRef(ortho);

      log("Leyendo metadata...");
      const metaRes = await fetch(`${API}/ortho-info?path=${encodeURIComponent(ortho.path)}`);
      if (metaRes.ok) {
        const meta = await metaRes.json();
        setOrthoMetadata(meta);
      }
      setUploadProgress((prev) => Math.max(prev, 90));

      log("Generando preview...");
      setOrthoImageUrl(`${API}/ortho-preview?path=${encodeURIComponent(ortho.path)}`);
      setWmsImageUrl(`${API}/ortho-wms-preview?path=${encodeURIComponent(ortho.path)}`);
      setBaseLayer("ortho");

      log("Carga finalizada.");
      setUploadProgress(100);
      setAppPhase("editor");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Error al subir archivos");
    } finally {
      setUploading(false);
    }
  };

  // ========== MAP INTERACTION FUNCTIONS (MINIMAP) ==========
  const handleMapWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!mapContainerRef.current) return;
    const rect = mapContainerRef.current.getBoundingClientRect();

    // Mouse position relative to container
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10, mapZoom * delta));

    if (newZoom !== mapZoom) {
      // Adjustment to mapOffset to zoom towards mouse position
      // Formula: offset = mouse_pos - (mouse_pos - old_offset) * (new_zoom / old_zoom)
      const ratio = newZoom / mapZoom;
      const newOffsetX = mouseX - (mouseX - mapOffset.x) * ratio;
      const newOffsetY = mouseY - (mouseY - mapOffset.y) * ratio;

      setMapZoom(newZoom);
      setMapOffset({ x: newOffsetX, y: newOffsetY });
    }
  }, [mapZoom, mapOffset]);

  const handleMapMouseDown = useCallback((e: ReactMouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (pickerMode) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - mapOffset.x, y: e.clientY - mapOffset.y });
  }, [pickerMode, mapOffset]);

  const handleMapMouseMove = useCallback((e: ReactMouseEvent) => {
    if (!isDragging) return;
    e.preventDefault();
    e.stopPropagation();
    setMapOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  }, [isDragging, dragStart]);

  const handleMapMouseUp = useCallback((e?: ReactMouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    setIsDragging(false);
  }, []);

  const handleMapClick = useCallback((e: ReactMouseEvent) => {
    if (!pickerMode || !orthoMetadata || !mapContainerRef.current) return;
    const rect = mapContainerRef.current.getBoundingClientRect();

    // 1. Obtener la posición del clic relativa al contenedor
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // 2. Restar el offset y dividir por zoom para obtener coordenada en el espacio "sin transformar" de la imagen
    // El offset aplica al grupo que contiene imagen y SVG
    const imageX = (clickX - mapOffset.x) / mapZoom;
    const imageY = (clickY - mapOffset.y) / mapZoom;

    // 3. Validar si el clic cayó dentro de la imagen
    if (imageX < 0 || imageX > mapSize.width || imageY < 0 || imageY > mapSize.height) return;

    const [minE, minN, maxE, maxN] = orthoMetadata.bounds;

    // 4. Proyección: Píxel -> Geográfico
    // E aumenta hacia derecha (X), N aumenta hacia arriba (Y disminuye)
    const pctX = imageX / mapSize.width;
    const pctY = imageY / mapSize.height;

    const E = minE + pctX * (maxE - minE);
    const N = maxN - pctY * (maxN - minN);

    if (editingPointId) {
      setRoutePoints((prev) =>
        prev.map((p) => {
          if (p.id !== editingPointId) return p;
          const updated = pickerMode === "start" ? { ...p, E_ini: E, N_ini: N } : { ...p, E_fin: E, N_fin: N };
          updated.heading = computeHeading(updated.E_ini, updated.N_ini, updated.E_fin, updated.N_fin);
          return updated;
        })
      );
      if (autoAdvancePicker && pickerMode === "start") {
        setPickerMode("end");
      } else {
        setPickerMode(null);
        setEditingPointId(null);
        setAutoAdvancePicker(false);
      }
    } else if (pickerMode === "start") {
      const prevEnd = routePoints.length > 0 ? routePoints[routePoints.length - 1].t_end : 0;
      const nextStart = clampTime(prevEnd);
      const nextEnd = clampTime(prevEnd + 10);
      const newPoint: RoutePoint = {
        id: generateId(), E_ini: E, N_ini: N, E_fin: E, N_fin: N,
        t_start: nextStart,
        t_end: nextEnd,
        heading: 0,
      };
      setRoutePoints((prev) => [...prev, newPoint]);
      setEditingPointId(newPoint.id);
      setPickerMode("end");
      setAutoAdvancePicker(true);
    }
  }, [pickerMode, orthoMetadata, mapOffset, mapZoom, mapSize, editingPointId, routePoints, autoAdvancePicker]);

  const addNewPoint = () => { setPickerMode("start"); setEditingPointId(null); setAutoAdvancePicker(true); };
  const startEditPoint = (id: string, mode: PickerMode) => {
    setEditingPointId(id);
    setPickerMode(mode);
    setAutoAdvancePicker(false);
  };
  const deletePoint = (id: string) => {
    setRoutePoints((prev) => prev.filter((p) => p.id !== id));
    if (editingPointId === id) {
      setEditingPointId(null);
      setPickerMode(null);
      setAutoAdvancePicker(false);
    }
  };
  const updatePointTime = (id: string, field: "t_start" | "t_end", value: number) => {
    const clamped = clampTime(value);
    setRoutePoints((prev) => prev.map((p) => (p.id === id ? { ...p, [field]: clamped } : p)));
    setTimeDrafts((prev) => ({
      ...prev,
      [id]: {
        t_start: field === "t_start" ? formatTime(clamped) : prev[id]?.t_start ?? "00:00",
        t_end: field === "t_end" ? formatTime(clamped) : prev[id]?.t_end ?? "00:00",
      },
    }));
  };
  const updateVectorInput = (index: number, updates: Partial<VectorInput>) => {
    setVectorInputs((prev) => prev.map((layer, i) => (i === index ? { ...layer, ...updates } : layer)));
  };

  const addVectorLayer = (label?: string) => {
    const newIndex = vectorInputs.length;
    const colorIndex = newIndex % DEFAULT_LAYER_COLORS.length;
    const newLayer: VectorInput = {
      id: generateId(),
      label: label || `Capa ${newIndex + 1}`,
      color: DEFAULT_LAYER_COLORS[colorIndex],
      width: 3,
      pattern: "solid",
      visible: true,
      file: null,
      ref: null,
      geometry: null,
      uploading: false,
      error: null,
    };
    setVectorInputs((prev) => [...prev, newLayer]);
  };

  const removeVectorLayer = (index: number) => {
    if (vectorInputs.length <= 1) return; // Keep at least one layer
    setVectorInputs((prev) => prev.filter((_, i) => i !== index));
  };

  const renameVectorLayer = (index: number, newLabel: string) => {
    updateVectorInput(index, { label: newLabel });
  };

  const toggleVectorLayer = (index: number) => {
    setVectorInputs((prev) =>
      prev.map((layer, i) => (i === index ? { ...layer, visible: !layer.visible } : layer))
    );
  };

  const reorderVectorLayers = (startIndex: number, endIndex: number) => {
    setVectorInputs((prev) => {
      const result = Array.from(prev);
      const [removed] = result.splice(startIndex, 1);
      result.splice(endIndex, 0, removed);
      return result;
    });
  };

  // ========== PATH FUNCTIONS ==========
  const togglePlayback = () => {
    const video = videoElement.current;
    if (!video) return;
    if (video.paused) video.play(); else video.pause();
  };

  const handleVideoClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!annotating || !videoElement.current) return;
    event.preventDefault();
    videoElement.current.pause();
    const rect = videoElement.current.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    const time = videoElement.current.currentTime;
    setPathPoints((prev) => [...prev, { t: time, x, y }]);
  };

  const startAnnotation = () => {
    videoElement.current?.pause();
    setAnnotating(true);
    setTrackData(null);
    setPathJobId(null);
  };

  const clearPoints = () => { setAnnotating(false); setPathPoints([]); setTrackData(null); };

  const pickSaveFile = async () => {
    if (!("showSaveFilePicker" in window)) { setPathError("Navegador no soportado"); return; }
    try {
      const showSavePicker = (window as Window & { showSaveFilePicker?: ShowSaveFilePicker }).showSaveFilePicker;
      if (!showSavePicker) { setPathError("Navegador no soportado"); return; }
      const handle = await showSavePicker({
        suggestedName: "overlay_lindero.mp4",
        types: [{ description: "Video MP4", accept: { "video/mp4": [".mp4"] } }],
      });
      setSaveHandle(handle);
      setSaveFileName(handle?.name ?? "overlay_lindero.mp4");
    } catch { setPathError("Error al seleccionar archivo"); }
  };

  const runTracking = async () => {
    if (!videoFile || pathPoints.length === 0) return;
    setAnnotating(false);
    setPathError(null);
    try {
      let ref = videoRef;
      if (!ref) { ref = await uploadFile(videoFile); setVideoRef(ref); }
      const res = await fetch(`${API}/track`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: ref.path, points: pathPoints, output_name: "track.json" }),
      });
      if (!res.ok) throw new Error("Fallo el tracking");
      const data = await res.json();
      setPathJobId(data.job_id);
      setPathJobStatus({ status: data.status, progress: 0, total: 0 });
    } catch (err) { setPathError(err instanceof Error ? err.message : "Error"); }
  };

  const renderOverlay = async () => {
    if (!videoRef || !trackPath) return;
    setPathError(null);
    try {
      const outName = saveFileName?.toLowerCase().endsWith(".mp4") ? saveFileName : `${saveFileName}.mp4`;
      const res = await fetch(`${API}/render-overlay`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: videoRef.path, track_path: trackPath, output_name: outName,
          line_color: pathLineColor, line_width: pathLineWidth, point_color: "#ffffff"
        }),
      });
      if (!res.ok) throw new Error("Fallo el render");
      const data = await res.json();
      setOverlayJobId(data.job_id);
    } catch (err) { setPathError(err instanceof Error ? err.message : "Error"); }
  };

  // ========== MINIMAP SAVE FUNCTIONS ==========
  const pickMinimapSaveFile = useCallback(async () => {
    if (!("showSaveFilePicker" in window)) {
      setMinimapSaveError("Navegador no soportado para guardado directo.");
      return;
    }
    try {
      const showSavePicker = (window as Window & { showSaveFilePicker?: ShowSaveFilePicker }).showSaveFilePicker;
      if (!showSavePicker) {
        setMinimapSaveError("Navegador no soportado para guardado directo.");
        return;
      }
      const handle = await showSavePicker({
        suggestedName: outputName.toLowerCase().endsWith(".mp4") ? outputName : `${outputName}.mp4`,
        types: [{ description: "Video MP4", accept: { "video/mp4": [".mp4"] } }],
      });
      setMinimapSaveHandle(handle);
      setMinimapSaveName(handle?.name ?? outputName);
      setOutputName(handle?.name ?? outputName);
      setMinimapSaveError(null);
      setMinimapSavedPath(null);
    } catch {
      setMinimapSaveError("Error al seleccionar archivo.");
    }
  }, [outputName]);

  const saveMinimapToDisk = useCallback(async (path: string) => {
    if (!minimapSaveHandle) return;
    if (minimapSavedPath === path) return;
    setMinimapSaving(true);
    try {
      const res = await fetch(`${API}/download?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error("No se pudo descargar el video.");
      const blob = await res.blob();
      const writable = await minimapSaveHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      setMinimapSavedPath(path);
      setMinimapSaveError(null);
    } catch (err) {
      setMinimapSaveError(err instanceof Error ? err.message : "Error al guardar el video.");
    } finally {
      setMinimapSaving(false);
    }
  }, [minimapSaveHandle, minimapSavedPath]);

  useEffect(() => {
    if (mode !== "minimap") return;
    if (!minimapExportPath || !minimapSaveHandle) return;
    void saveMinimapToDisk(minimapExportPath);
  }, [mode, minimapExportPath, minimapSaveHandle, saveMinimapToDisk]);

  // Real-time preview effect
  useEffect(() => {
    if (routePoints.length === 0 || previewBusy) return;
    // Debounce preview update
    const timer = setTimeout(() => {
      handlePreview();
    }, 400); // 400ms debounce
    return () => clearTimeout(timer);
  }, [previewTime, orthoRef, routePoints.length, mapZoomFactor, mapHalfWidth, arrowSize, coneAngle, coneLength, coneOpacity, iconCircleOpacity, iconCircleSize, showCompass, compassSize]);

  // ========== MINIMAP RENDER FUNCTIONS ==========
  const handlePreview = async () => {
    // Si no hay orto o ruta, igual intentamos renderizar si tenemos baseLayer=wms
    if (routePoints.length === 0) return;

    // Si baseLayer es ortho pero no hay orto, requerir orto
    if (baseLayer === "ortho" && !orthoRef) return;

    setPreviewBusy(true);
    try {
      const safeOutputName = outputName.toLowerCase().endsWith(".mp4") ? outputName : `${outputName}.mp4`;
      const effectiveMapHalfWidth = mapHalfWidth / Math.max(0.01, mapZoomFactor);

      // Si tenemos orto, llamamos al backend para el preview "Call of Duty style"
      if (orthoRef) {
        const csvBlob = new Blob([csvContent], { type: "text/csv" });
        const csvFileRef = await uploadFile(new File([csvBlob], "route.csv"));

        const config: RenderConfig = {
          ortho_path: orthoRef.path,
          csv_path: csvFileRef.path,
          vectors_paths: activeVectorLayers.map((layer) => layer.path),
          vector_layers: activeVectorLayers,
          fps, duration_sec: duration, width, height, map_half_width_m: effectiveMapHalfWidth, arrow_size_px: arrowSize, cone_angle_deg: coneAngle, cone_length_px: coneLength, cone_opacity: coneOpacity,
          icon_circle_opacity: iconCircleOpacity,
          icon_circle_size_px: iconCircleSize,
          show_compass: showCompass,
          compass_size_px: compassSize,
          line_color: DEFAULT_LINE_COLOR, line_width: DEFAULT_LINE_WIDTH, boundary_color: DEFAULT_BOUNDARY_COLOR, boundary_width: DEFAULT_BOUNDARY_WIDTH, point_color: DEFAULT_POINT_COLOR, output_name: safeOutputName, use_gpu: useGpu, workers: 0
        };

        const res = await fetch(`${API}/preview`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config, time_sec: previewTime }) });
        if (!res.ok) {
          const details = await res.text().catch(() => "");
          throw new Error(`Preview failed (${res.status}) ${details}`);
        }
        setPreviewUrl(URL.createObjectURL(await res.blob()));
      } else {
        // En modo satelital puro, no llamamos al backend para el render "preview" 
        // (porque backend requiere dataset). 
        // Podríamos generar una imagen estática con WMS pero por ahora limpiamos previewUrl
        // para mostrar el placeholder o el mapa baseWMS si implementamos logica extra.
        // Ojo: El user pidió "mode planteo".
        setPreviewUrl(null);
      }
      setPreviewError(null);
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : "Error al generar vista previa");
    } finally {
      setPreviewBusy(false);
    }
  };


  const handleRender = async () => {
    if (routePoints.length === 0) return;

    // El renderizado final REQUIERE ortomosaico porque backend/render.py lo usa
    if (!orthoRef) {
      alert("Para renderizar el video final, debes cargar un ortomosaico GeoTIFF. El modo satelital es solo para planteo/diseño.");
      return;
    }

    if ("showSaveFilePicker" in window && !minimapSaveHandle) {
      setMinimapSaveError("Selecciona destino antes de renderizar.");
      return;
    }
    setMinimapSaveError(null);
    setShowVideoModal(false);
    setMinimapExportPath(null);
    setMinimapSavedPath(null);
    setJobStatus(null);

    try {
      const safeOutputName = outputName.toLowerCase().endsWith(".mp4") ? outputName : `${outputName}.mp4`;
      const effectiveMapHalfWidth = mapHalfWidth / Math.max(0.01, mapZoomFactor);
      const csvBlob = new Blob([csvContent], { type: "text/csv" });
      const csvFileRef = await uploadFile(new File([csvBlob], "route.csv"));

      const config: RenderConfig = {
        ortho_path: orthoRef.path,
        csv_path: csvFileRef.path,
        vectors_paths: activeVectorLayers.map((layer) => layer.path),
        vector_layers: activeVectorLayers,
        fps, duration_sec: duration, width, height, map_half_width_m: effectiveMapHalfWidth, arrow_size_px: arrowSize, cone_angle_deg: coneAngle, cone_length_px: coneLength, cone_opacity: coneOpacity,
        icon_circle_opacity: iconCircleOpacity,
        icon_circle_size_px: iconCircleSize,
        show_compass: showCompass,
        compass_size_px: compassSize,
        line_color: DEFAULT_LINE_COLOR, line_width: DEFAULT_LINE_WIDTH, boundary_color: DEFAULT_BOUNDARY_COLOR, boundary_width: DEFAULT_BOUNDARY_WIDTH, point_color: DEFAULT_POINT_COLOR, output_name: safeOutputName, use_gpu: useGpu, workers: 0
      };

      const res = await fetch(`${API}/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
      if (!res.ok) {
        const details = await res.text().catch(() => "");
        throw new Error(`Render failed (${res.status}) ${details}`);
      }
      const data = await res.json();
      setJobId(data.job_id);
    } catch (err) { console.error(err); }
  };

  const handleSaveProject = () => {
    const project = {
      version: "1.0",
      timestamp: new Date().toISOString(),
      settings: {
        fps, width, height, mapHalfWidth, mapZoomFactor, arrowSize, iconCircleSize,
        coneAngle, coneLength, coneOpacity, iconCircleOpacity, showCompass, compassSize,
        outputName
      },
      ortho: {
        ref: orthoRef,
        metadata: orthoMetadata,
        imageUrl: orthoImageUrl,
        wmsUrl: wmsImageUrl,
        baseLayer,
        naturalSize
      },
      routePoints,
      vectorInputs: vectorInputs.map(v => ({
        id: v.id, label: v.label, color: v.color, width: v.width, pattern: v.pattern,
        visible: v.visible, ref: v.ref, geometry: v.geometry
      }))
    };
    const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `proyecto_${outputName.replace(".mp4", "") || "minimapa"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoadProject = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const project = JSON.parse(ev.target?.result as string);
        if (project.settings) {
          setFps(project.settings.fps);
          setWidth(project.settings.width);
          setHeight(project.settings.height);
          setMapHalfWidth(project.settings.mapHalfWidth);
          setMapZoomFactor(project.settings.mapZoomFactor);
          setArrowSize(project.settings.arrowSize);
          setIconCircleSize(project.settings.iconCircleSize);
          setConeAngle(project.settings.coneAngle);
          setConeLength(project.settings.coneLength);
          setConeOpacity(project.settings.coneOpacity);
          setIconCircleOpacity(project.settings.iconCircleOpacity);
          setShowCompass(project.settings.showCompass);
          setCompassSize(project.settings.compassSize);
          setOutputName(project.settings.outputName);
        }
        if (project.ortho) {
          setOrthoRef(project.ortho.ref);
          setOrthoMetadata(project.ortho.metadata);
          setOrthoImageUrl(project.ortho.imageUrl);
          setWmsImageUrl(project.ortho.wmsUrl);
          setBaseLayer(project.ortho.baseLayer);
          setNaturalSize(project.ortho.naturalSize);
        }
        if (project.routePoints) setRoutePoints(project.routePoints);
        if (project.vectorInputs) {
          setVectorInputs(project.vectorInputs.map((v: any) => ({
            ...v, file: null, uploading: false, error: null
          })));
        }
        setAppPhase("editor");
      } catch (err) {
        alert("Error al cargar el proyecto: Archivo inválido.");
      }
    };
    reader.readAsText(file);
    e.target.value = ""; // Reset input
  };

  const handleCleanup = async (allData = false) => {
    try {
      const endpoint = allData ? `${API}/cleanup/data` : `${API}/cleanup`;
      const res = await fetch(endpoint, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        alert(data.message);
      }
    } catch (err) {
      console.error(err);
      alert("Error al limpiar archivos.");
    }
  };

  const coordToPixel = useCallback((E: number, N: number) => {
    if (!orthoMetadata || !mapSize.width) return { x: 0, y: 0 };
    const [minE, minN, maxE, maxN] = orthoMetadata.bounds;

    // Proporción de la coordenada dentro de los límites geográficos
    const pctE = (E - minE) / (maxE - minE);
    const pctN = (maxN - N) / (maxN - minN);

    // Convertir a píxeles basados en el tamaño de renderizado del mapa
    const x = pctE * mapSize.width;
    const y = pctN * mapSize.height;
    return { x, y };
  }, [orthoMetadata, mapSize]);

  const vectorSvgData = useMemo(() => {
    if (!orthoMetadata || mapSize.width === 0 || mapSize.height === 0) return [];
    return vectorInputs
      .filter((layer) => layer.geometry && (layer.geometry.paths.length || layer.geometry.points.length))
      .map((layer) => ({
        id: layer.id,
        color: layer.color,
        width: layer.width,
        pattern: layer.pattern,
        paths: layer.geometry!.paths.map((path) => buildPathD(path, coordToPixel)),
        points: layer.geometry!.points.map((pt) => coordToPixel(pt[0], pt[1])),
      }));
  }, [vectorInputs, orthoMetadata, mapSize, coordToPixel]);

  // ==================== RENDERING ====================
  return (
    <div className={`min-h-screen ${mode === "path" ? "path-theme" : "minimap-theme"}`}>
      {/* Header */}
      <header className="px-6 py-6 md:px-10">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-[var(--text)] md:text-4xl">
                {mode === "path" ? "Generador de path aéreo" : "Generador de minimapas"}
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMode(mode === "minimap" ? "path" : "minimap")}
                className="relative flex h-10 w-48 items-center overflow-hidden rounded-full border border-[var(--line)] bg-[var(--panel)]/80 px-2 text-xs font-medium text-[var(--text)] shadow-sm transition-all hover:shadow-md"
              >
                <span className="relative z-10 w-full text-center">
                  {mode === "path" ? "Modo path aéreo" : "Modo minimapa"}
                </span>
                <span className={`absolute top-1 h-8 w-8 rounded-full bg-[var(--accent)] shadow-lg transition-all ${mode === "path" ? "right-1" : "left-1"}`} />
              </button>
              <div className="flex items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--panel)]/70 px-2 py-1 shadow-sm">
                <button
                  onClick={handleSaveProject}
                  className="rounded-full px-3 py-1.5 text-xs font-semibold text-emerald-700 hover:bg-emerald-50 transition"
                  disabled={false} // Always allow saving project metadata
                  title="Guardar estado del proyecto actual"
                >
                  💾 Guardar
                </button>
                <label className="cursor-pointer rounded-full px-3 py-1.5 text-xs font-semibold text-blue-700 hover:bg-blue-50 transition" title="Abrir un archivo de proyecto (.json)">
                  📂 Abrir
                  <input type="file" accept=".json" onChange={handleLoadProject} className="hidden" />
                </label>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--panel)]/70 px-4 py-2.5 text-sm shadow-sm">
                <span className={`h-2.5 w-2.5 rounded-full ${backendOnline ? "bg-emerald-500" : "bg-red-500"}`} />
                {backendOnline ? "Backend OK" : "Backend OFF"}
              </div>
              <div className="hidden rounded-full border border-[var(--line)] bg-[var(--panel)]/70 px-4 py-2.5 text-sm shadow-sm sm:block">
                EPSG:9377 Colombia
              </div>
            </div>
          </div>
        </div>
      </header>

      {mode === "minimap" ? (
        // ==================== MINIMAP MODE ====================
        appPhase === "upload" ? (
          <main className="mx-auto max-w-3xl px-6 pb-16">
            <div className="rounded-3xl border border-[var(--line)] bg-[var(--panel)] p-8 shadow-xl">
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-[var(--text)]">Entradas principales</h2>
                <p className="mt-1 text-sm text-[var(--muted)]">Carga el ortomosaico para comenzar</p>
              </div>
              <div className="space-y-5">
                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-[var(--text)]">Ortomosaico GeoTIFF (Opcional)</span>
                  <input type="file" accept=".tif,.tiff" onChange={(e) => setOrthoFile(e.target.files?.[0] ?? null)} className="w-full rounded-xl border border-[var(--line)] bg-[var(--bg-muted)] px-4 py-3.5 text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-[var(--accent)] file:px-4 file:py-2 file:text-white" />
                  {orthoFile && <span className="mt-2 block text-xs text-[var(--accent)]">✓ {orthoFile.name}</span>}
                </label>
                {uploadError && <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-sm text-red-600">{uploadError}</div>}
                <button onClick={handleUploadAndStart} disabled={uploading} className="mt-4 w-full rounded-full bg-gradient-to-r from-[var(--accent)] to-blue-600 py-4 font-semibold text-white shadow-lg disabled:opacity-50">
                  {uploading ? "Cargando..." : "Iniciar minimapa"}
                </button>
                <div className="mt-4 flex flex-wrap justify-center gap-3">
                  <button
                    onClick={() => handleCleanup(false)}
                    className="rounded-full border border-orange-500/30 bg-orange-50/50 px-3 py-1.5 text-[10px] font-medium text-orange-700 hover:bg-orange-100 transition"
                  >
                    Limpiar Cache
                  </button>
                  <button
                    onClick={() => {
                      if (confirm("¿Borrar todos los archivos subidos? Tendrás que subir ortos y vectores de nuevo.")) {
                        handleCleanup(true);
                      }
                    }}
                    className="rounded-full border border-red-500/30 bg-red-50/50 px-3 py-1.5 text-[10px] font-medium text-red-700 hover:bg-red-100 transition"
                  >
                    Limpiar Todo
                  </button>
                </div>
                {(uploading || uploadProgress > 0) && (
                  <div className="mt-6 rounded-2xl border border-[var(--line)] bg-white/70 p-4">
                    <div className="flex items-center justify-between text-xs text-[var(--muted)]">
                      <span>Progreso</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full bg-[var(--accent)] transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <div className="mt-3 space-y-1 text-xs text-[var(--muted)]">
                      {uploadLogs.map((line, idx) => (
                        <div key={`${line}-${idx}`}>• {line}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </main>
        ) : (
          <main className="mx-auto max-w-[1800px] px-4 pb-8">
            <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
              <div className="flex flex-col gap-4">
                <section className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-lg">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-lg font-semibold text-[var(--text)]">Selector de puntos (Posición)</h2>
                    <div className="flex items-center gap-2 rounded-full border border-[var(--line)] bg-white/70 p-1 text-xs">
                      <button
                        onClick={() => setBaseLayer("ortho")}
                        className={`rounded-full px-3 py-1 ${baseLayer === "ortho" ? "bg-[var(--accent)] text-white" : "text-[var(--muted)]"}`}
                      >
                        Ortomosaico
                      </button>
                      <button
                        onClick={() => setBaseLayer("wms")}
                        className={`rounded-full px-3 py-1 ${baseLayer === "wms" ? "bg-[var(--accent)] text-white" : "text-[var(--muted)]"}`}
                      >
                        Satelital
                      </button>
                    </div>
                  </div>
                  {pickerMode && <div className="mb-3 flex items-center gap-2 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm">Punto {pickerLabel} <button onClick={() => { setPickerMode(null); setEditingPointId(null); setAutoAdvancePicker(false); }} className="ml-auto text-xs text-amber-600">Cancelar</button></div>}
                  <div ref={mapContainerRef} className="relative h-[600px] w-full overflow-hidden rounded-xl border border-[var(--line)] bg-[#0f172a] shadow-inner cursor-crosshair overscroll-contain touch-none" onWheel={handleMapWheel} onMouseDown={handleMapMouseDown} onMouseMove={handleMapMouseMove} onMouseUp={handleMapMouseUp} onMouseLeave={handleMapMouseUp} onClick={handleMapClick}>

                    {/* Contenedor transformable que agrupa Imagen + SVG */}
                    <div
                      style={{
                        transform: `translate(${mapOffset.x}px, ${mapOffset.y}px) scale(${mapZoom})`,
                        transformOrigin: "0 0",
                        width: mapSize.width > 0 ? mapSize.width : 'auto',
                        height: mapSize.height > 0 ? mapSize.height : 'auto'
                      }}
                      className="origin-top-left"
                    >
                      {(baseLayer === "wms" || !orthoImageUrl) && wmsImageUrl && (
                        <img
                          src={wmsImageUrl}
                          alt="Satellite Base"
                          draggable={false}
                          className="select-none pointer-events-none absolute inset-0 block h-full w-full object-cover"
                          style={{ zIndex: 0 }}
                        />
                      )}

                      {orthoImageUrl && (
                        <img
                          ref={mapImageRef}
                          src={orthoImageUrl}
                          alt="Ortho Overlay"
                          draggable={false}
                          onLoad={(e) => {
                            const img = e.currentTarget;
                            if (orthoRef) setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
                          }}
                          className="select-none pointer-events-none absolute inset-0 block h-full w-full object-contain"
                          style={{ zIndex: 10, opacity: 0.8 }} // Overlay with opacity
                        />
                      )}

                      {/* Capa de vectores SVG superpuesta exactamente sobre la imagen */}
                      <svg
                        className="pointer-events-none absolute inset-0 left-0 top-0 overflow-visible"
                        width={mapSize.width}
                        height={mapSize.height}
                        style={{ zIndex: 20 }}
                        viewBox={`0 0 ${mapSize.width} ${mapSize.height}`}
                      >
                        {vectorSvgData.map((layer) => {
                          const dash = dashArrayForPattern(layer.pattern, vectorScale);
                          const strokeW = Math.max(1, layer.width) * vectorScale;
                          const pointR = Math.max(2, strokeW + 1);
                          return (
                            <g key={layer.id} opacity={0.95}>
                              {layer.paths.map((d, idx) => (
                                <path
                                  key={`${layer.id}-p-${idx}`}
                                  d={d}
                                  fill="none"
                                  stroke={layer.color}
                                  strokeWidth={strokeW}
                                  strokeDasharray={dash}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              ))}
                              {layer.points.map((pt, idx) => (
                                <circle key={`${layer.id}-pt-${idx}`} cx={pt.x} cy={pt.y} r={pointR} fill={layer.color} />
                              ))}
                            </g>
                          );
                        })}
                        {routePoints.map((p, i) => {
                          const s = coordToPixel(p.E_ini, p.N_ini);
                          const e = coordToPixel(p.E_fin, p.N_fin);
                          const strokeW = 3 / mapZoom;
                          const r = 10 / mapZoom;
                          const fontSize = 14 / mapZoom;

                          return (
                            <g key={p.id}>
                              {/* Línea conectora */}
                              <line x1={s.x} y1={s.y} x2={e.x} y2={e.y} stroke="#3b82f6" strokeWidth={strokeW} strokeDasharray={`${8 / mapZoom} ${4 / mapZoom}`} strokeLinecap="round" />

                              {/* Punto Inicio */}
                              <circle cx={s.x} cy={s.y} r={r} fill="#22c55e" stroke="white" strokeWidth={2 / mapZoom} />
                              <text x={s.x} y={s.y - 15 / mapZoom} textAnchor="middle" fontSize={fontSize} fontWeight="bold" fill="white" style={{ textShadow: "0px 1px 3px rgba(0,0,0,0.8)" }}>{getPointLabel(i, "start")}</text>

                              {/* Punto Fin */}
                              <circle cx={e.x} cy={e.y} r={r} fill="#ef4444" stroke="white" strokeWidth={2 / mapZoom} />
                              <text x={e.x} y={e.y - 15 / mapZoom} textAnchor="middle" fontSize={fontSize} fontWeight="bold" fill="white" style={{ textShadow: "0px 1px 3px rgba(0,0,0,0.8)" }}>{getPointLabel(i, "end")}</text>
                            </g>
                          );
                        })}
                      </svg>
                    </div>
                  </div>
                  <div className="mt-3 flex gap-2">
                    <button
                      onClick={addNewPoint}
                      disabled={isPicking}
                      className={`rounded-lg px-4 py-2 text-sm font-medium text-white transition ${isPicking ? "bg-red-500" : "bg-[var(--accent)]"
                        } ${isPicking ? "opacity-80" : ""}`}
                    >
                      {isPicking ? "Seleccionando A/B..." : "Agregar segmento"}
                    </button>
                  </div>
                </section>
                <section className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-lg overflow-x-auto">

                  <table className="w-full text-sm">

                    <thead>

                      <tr className="border-b text-left text-xs text-[var(--muted)]">

                        <th className="px-2 py-2">Segmento</th>

                        <th className="px-2 py-2">Inicio (A)</th>

                        <th className="px-2 py-2">Fin (B)</th>

                        <th className="px-2 py-2">t_start (mm:ss)</th>

                        <th className="px-2 py-2">t_end (mm:ss)</th>

                        <th className="px-2 py-2">Heading</th>

                        <th className="px-2 py-2">Editar</th>

                        <th className="px-2 py-2"></th>

                      </tr>

                    </thead>

                    <tbody>

                      {routePoints.map((p, i) => {
                        const isEditing = p.id === editingPointId;
                        const draft = timeDrafts[p.id] ?? { t_start: formatTime(p.t_start), t_end: formatTime(p.t_end) };
                        return (

                          <tr

                            key={p.id}

                            className={`border-b hover:bg-[var(--bg-muted)] ${isEditing ? "bg-amber-50/60" : ""}`}

                          >

                            <td className="px-2 py-2 text-xs font-semibold text-[var(--text)]">

                              {getPointLabel(i, "start")}/{getPointLabel(i, "end")}

                            </td>

                            <td className="px-2 py-2 text-[11px]">

                              <div className="mono text-[11px]">E {p.E_ini.toFixed(2)}</div>

                              <div className="mono text-[11px]">N {p.N_ini.toFixed(2)}</div>

                            </td>

                            <td className="px-2 py-2 text-[11px]">

                              <div className="mono text-[11px]">E {p.E_fin.toFixed(2)}</div>

                              <div className="mono text-[11px]">N {p.N_fin.toFixed(2)}</div>

                            </td>

                            <td className="px-2 py-2">

                              <input

                                type="text"

                                inputMode="numeric"

                                value={draft.t_start}

                                placeholder="00:00"

                                onChange={(e) => {
                                  const value = e.target.value;
                                  if (/^\d{0,2}(:\d{0,2})?$/.test(value)) {
                                    setTimeDrafts((prev) => ({
                                      ...prev,
                                      [p.id]: { ...draft, t_start: value },
                                    }));
                                  }
                                }}

                                onBlur={() => {
                                  const parsed = parseTimeInput(draft.t_start);
                                  if (parsed !== null) {
                                    updatePointTime(p.id, "t_start", parsed);
                                  } else {
                                    setTimeDrafts((prev) => ({
                                      ...prev,
                                      [p.id]: { ...draft, t_start: formatTime(p.t_start) },
                                    }));
                                  }
                                }}

                                className="w-20 rounded border px-1 py-0.5 text-xs"

                              />

                            </td>

                            <td className="px-2 py-2">

                              <input

                                type="text"

                                inputMode="numeric"

                                value={draft.t_end}

                                placeholder="00:00"

                                onChange={(e) => {
                                  const value = e.target.value;
                                  if (/^\d{0,2}(:\d{0,2})?$/.test(value)) {
                                    setTimeDrafts((prev) => ({
                                      ...prev,
                                      [p.id]: { ...draft, t_end: value },
                                    }));
                                  }
                                }}

                                onBlur={() => {
                                  const parsed = parseTimeInput(draft.t_end);
                                  if (parsed !== null) {
                                    updatePointTime(p.id, "t_end", parsed);
                                  } else {
                                    setTimeDrafts((prev) => ({
                                      ...prev,
                                      [p.id]: { ...draft, t_end: formatTime(p.t_end) },
                                    }));
                                  }
                                }}

                                className="w-20 rounded border px-1 py-0.5 text-xs"

                              />

                            </td>

                            <td className="px-2 py-2 text-xs">{p.heading.toFixed(1)}?</td>

                            <td className="px-2 py-2">

                              <div className="flex flex-wrap gap-2">

                                <button

                                  onClick={() => startEditPoint(p.id, "start")}

                                  className="rounded-full border border-[var(--line)] bg-white/70 px-3 py-1 text-[11px] text-[var(--text)]"

                                >

                                  Pick A

                                </button>

                                <button

                                  onClick={() => startEditPoint(p.id, "end")}

                                  className="rounded-full border border-[var(--line)] bg-white/70 px-3 py-1 text-[11px] text-[var(--text)]"

                                >

                                  Pick B

                                </button>

                              </div>

                            </td>

                            <td className="px-2 py-2 text-right">

                              <button onClick={() => deletePoint(p.id)} className="text-red-500">

                                x

                              </button>

                            </td>

                          </tr>

                        );

                      })}

                    </tbody>

                  </table>

                </section>

                {/* LOG CONSOLE & SYSTEM STATS */}

              </div>
              <div className="flex flex-col gap-4">
                <section className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-lg">
                  <h2 className="mb-3 text-lg font-semibold text-[var(--text)]">Vista previa</h2>
                  <div
                    className="w-full mx-auto rounded-xl bg-[var(--bg-muted)] overflow-hidden shadow-inner flex items-center justify-center transition-all duration-300 aspect-video max-h-[600px]"
                  >
                    {previewUrl ? (
                      <img src={previewUrl} alt="Vista previa del minimapa" className="h-full w-full object-contain" />
                    ) : (
                      <div className="flex flex-col items-center gap-2 text-xs text-[var(--muted)]">
                        <span className="text-2xl opacity-20">🖼️</span>
                        <span>Sin vista previa</span>
                        <span className="opacity-50 text-[10px]">{width} x {height}</span>
                      </div>
                    )}
                  </div>
                  <div className="mt-4 flex items-center gap-3"><input type="range" min={0} max={Math.max(duration, 1)} step={0.1} value={previewTime} onChange={(e) => setPreviewTime(Number(e.target.value))} className="flex-1" /><span className="text-xs">{formatTime(previewTime)}</span></div>
                  <button onClick={handlePreview} disabled={previewBusy || routePoints.length === 0} className="mt-3 w-full rounded-xl bg-[var(--accent)] py-3 text-white font-semibold">{previewBusy ? "Generando..." : "Actualizar vista"}</button>
                  {previewError && <div className="mt-2 text-xs text-red-500 text-center">{previewError}</div>}
                </section>
              </div>
            </div>
            <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
              <div className="flex flex-col gap-4">
                <section className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-lg">
                  <h2 className="mb-3 text-lg font-semibold text-[var(--text)]">Ajustes</h2>
                  <div className="grid gap-3 text-sm md:grid-cols-2 xl:grid-cols-3">
                    <div className="col-span-full grid grid-cols-4 gap-2">
                      <label className="flex flex-col text-xs">FPS
                        <input type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} className="rounded border p-1 w-full" />
                      </label>
                      <label className="flex flex-col text-xs">Calidad
                        <select
                          className="rounded border p-1 w-full text-xs"
                          value={(() => {
                            if (width === 640 && height === 360) return "nHD";
                            if (width === 854 && height === 480) return "SD";
                            if (width === 1280 && height === 720) return "HD";
                            if (width === 1920 && height === 1080) return "FHD";
                            if (width === 2560 && height === 1440) return "2K";
                            if (width === 3840 && height === 2160) return "4K";
                            return "custom";
                          })()}
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val === "nHD") { setWidth(640); setHeight(360); }
                            if (val === "SD") { setWidth(854); setHeight(480); }
                            if (val === "HD") { setWidth(1280); setHeight(720); }
                            if (val === "FHD") { setWidth(1920); setHeight(1080); }
                            if (val === "2K") { setWidth(2560); setHeight(1440); }
                            if (val === "4K") { setWidth(3840); setHeight(2160); }
                          }}
                        >
                          <option value="custom">Custom</option>
                          <option value="nHD">nHD</option>
                          <option value="SD">SD</option>
                          <option value="HD">HD</option>
                          <option value="FHD">FHD</option>
                          <option value="2K">2K</option>
                          <option value="4K">4K</option>
                        </select>
                      </label>
                      <label className="flex flex-col text-xs">W (px)
                        <input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value))} className="rounded border p-1 w-full" />
                      </label>
                      <label className="flex flex-col text-xs">H (px)
                        <input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value))} className="rounded border p-1 w-full" />
                      </label>
                    </div>
                    <label className="flex flex-col">Mitad mapa (m)
                      <input type="number" value={mapHalfWidth} onChange={(e) => setMapHalfWidth(Number(e.target.value))} className="rounded border p-1" />
                    </label>
                    <label className="flex flex-col">Zoom ({mapZoomFactor.toFixed(2)}x)
                      <input type="range" min={0.01} max={5} step={0.01} value={mapZoomFactor} onChange={(e) => setMapZoomFactor(Number(e.target.value))} className="accent-[var(--accent)]" />
                    </label>
                    <label className="flex flex-col">Tamano icono (px)
                      <input type="number" value={arrowSize} onChange={(e) => setArrowSize(Number(e.target.value))} className="rounded border p-1" />
                    </label>
                    <label className="flex flex-col">Tamano circulo icono (px)
                      <input type="number" value={iconCircleSize} onChange={(e) => setIconCircleSize(Number(e.target.value))} className="rounded border p-1" />
                    </label>
                    <label className="flex flex-col">Angulo cono (deg)
                      <input type="number" value={coneAngle} onChange={(e) => setConeAngle(Number(e.target.value))} className="rounded border p-1" />
                    </label>
                    <label className="flex flex-col">Largo cono (px)
                      <input type="number" value={coneLength} onChange={(e) => setConeLength(Number(e.target.value))} className="rounded border p-1" />
                    </label>
                    <label className="flex flex-col">Opacidad radar ({coneOpacity.toFixed(2)})
                      <input type="range" min={0} max={1} step={0.05} value={coneOpacity} onChange={(e) => setConeOpacity(Number(e.target.value))} className="accent-[var(--accent)]" />
                    </label>
                    <label className="flex flex-col">Opacidad circulo icono ({iconCircleOpacity.toFixed(2)})
                      <input type="range" min={0} max={1} step={0.05} value={iconCircleOpacity} onChange={(e) => setIconCircleOpacity(Number(e.target.value))} className="accent-[var(--accent)]" />
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={showCompass} onChange={(e) => setShowCompass(e.target.checked)} className="accent-[var(--accent)]" />
                      Mostrar brújula
                    </label>
                    {showCompass && (
                      <label className="flex flex-col">Tamaño brújula ({compassSize}px)
                        <input type="range" min={20} max={100} step={5} value={compassSize} onChange={(e) => setCompassSize(Number(e.target.value))} className="accent-[var(--accent)]" />
                      </label>
                    )}
                    <div className="col-span-full border-t border-[var(--line)] pt-2 mt-1">
                      <label className="flex items-center justify-between gap-2 cursor-pointer bg-white/50 p-2 rounded-lg hover:bg-white/80 transition">
                        <span className="flex flex-col">
                          <span className="font-semibold text-[var(--accent)] text-xs">Modo de Procesamiento</span>
                          <span className="text-[10px] text-[var(--muted)]">{useGpu ? "GPU (Rápido, Nvidia)" : "CPU (Compatibilidad)"}</span>
                        </span>
                        <div className="relative">
                          <input type="checkbox" className="sr-only" checked={useGpu} onChange={(e) => setUseGpu(e.target.checked)} />
                          <div className={`block h-6 w-10 rounded-full transition-colors ${useGpu ? "bg-emerald-500" : "bg-slate-300"}`}></div>
                          <div className={`absolute left-1 top-1 h-4 w-4 rounded-full bg-white transition-transform ${useGpu ? "translate-x-4" : "translate-x-0"}`}></div>
                        </div>
                      </label>
                    </div>
                  </div>
                </section>
                <section className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-lg">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-semibold text-[var(--text)]">Ajustes de vectores</h2>
                    <button
                      onClick={() => {
                        const name = prompt("Nombre de la nueva capa:", `Capa ${vectorInputs.length + 1}`);
                        if (name) addVectorLayer(name);
                      }}
                      className="rounded-full border border-[var(--line)] bg-emerald-500 px-3 py-1 text-xs text-white hover:bg-emerald-600 transition"
                    >
                      + Añadir capa
                    </button>
                  </div>
                  <div className="space-y-3 text-sm">
                    {vectorInputs.map((layer, idx) => (
                      <div
                        key={layer.id}
                        draggable
                        onDragStart={(e) => e.dataTransfer.setData("text/plain", idx.toString())}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                          e.preventDefault();
                          const fromIdx = parseInt(e.dataTransfer.getData("text/plain"));
                          reorderVectorLayers(fromIdx, idx);
                        }}
                        className={`rounded-xl border border-[var(--line)] bg-[var(--bg-muted)] p-3 transition-opacity ${layer.visible ? "opacity-100" : "opacity-60 grayscale"}`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <div className="cursor-grab text-[var(--muted)] hover:text-[var(--text)]" title="Arrastrar para reordenar">
                              ⠿
                            </div>
                            <input
                              type="text"
                              value={layer.label}
                              onChange={(e) => renameVectorLayer(idx, e.target.value)}
                              className="text-sm font-semibold text-[var(--text)] bg-transparent border-b border-transparent hover:border-[var(--line)] focus:border-[var(--accent)] outline-none px-1"
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => toggleVectorLayer(idx)}
                              className={`text-sm px-2 transition-colors ${layer.visible ? "text-emerald-500" : "text-[var(--muted)]"}`}
                              title={layer.visible ? "Ocultar capa" : "Mostrar capa"}
                            >
                              {layer.visible ? "👁" : "👁‍🗨"}
                            </button>
                            <div className={`text-xs ${layer.ref ? "text-emerald-600" : "text-[var(--muted)]"}`}>
                              {layer.uploading ? "Subiendo..." : layer.ref ? "Cargado" : "Sin capa"}
                            </div>
                            {vectorInputs.length > 1 && (
                              <button
                                onClick={() => removeVectorLayer(idx)}
                                className="text-red-500 hover:text-red-700 text-xs px-2"
                                title="Eliminar capa"
                              >
                                ✕
                              </button>
                            )}
                          </div>
                        </div>
                        <label className="mt-2 block text-xs text-[var(--muted)]">Capa (GeoJSON)</label>
                        <input
                          type="file"
                          accept=".geojson,.json"
                          onChange={(e) => handleVectorFileChange(idx, e.target.files?.[0] ?? null)}
                          className="mt-2 w-full rounded-xl border border-[var(--line)] bg-white/80 px-4 py-3 text-xs file:mr-4 file:rounded-lg file:border-0 file:bg-slate-200 file:px-4"
                        />
                        {(layer.file || layer.ref) && (
                          <div className="mt-2 text-xs text-[var(--muted)]">
                            {layer.file ? layer.file.name : (layer.ref as any)?.filename || "Capa vinculada"}
                          </div>
                        )}
                        {layer.error && (
                          <div className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
                            {layer.error}
                          </div>
                        )}
                        {(layer.file || layer.ref) && (
                          <div className="mt-3 grid gap-3 text-xs md:grid-cols-3">
                            <label className="flex flex-col gap-1 text-[11px] text-[var(--muted)]">
                              Color
                              <input
                                type="color"
                                value={layer.color}
                                onChange={(e) => updateVectorInput(idx, { color: e.target.value })}
                                className="h-8 w-full rounded-lg border border-[var(--line)] bg-white"
                              />
                            </label>
                            <label className="flex flex-col gap-1 text-[11px] text-[var(--muted)]">
                              Grosor
                              <input
                                type="number"
                                min={1}
                                max={12}
                                value={layer.width}
                                onChange={(e) => updateVectorInput(idx, { width: Number(e.target.value) })}
                                className="w-full rounded-lg border border-[var(--line)] bg-white px-2 py-1 text-xs"
                              />
                            </label>
                            <label className="flex flex-col gap-1 text-[11px] text-[var(--muted)]">
                              Estilo
                              <select
                                value={layer.pattern}
                                onChange={(e) => updateVectorInput(idx, { pattern: e.target.value })}
                                className="w-full rounded-lg border border-[var(--line)] bg-white px-2 py-1 text-xs"
                              >
                                {LINE_STYLES.map((style) => (
                                  <option key={style.value} value={style.value}>
                                    {style.label}
                                  </option>
                                ))}
                              </select>
                            </label>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              </div>
              <section className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-lg">
                <h2 className="mb-3 text-lg font-semibold text-[var(--text)]">Exportar</h2>
                <div className="space-y-3 text-sm">
                  <label className="flex flex-col">Nombre archivo
                    <input type="text" value={outputName} onChange={(e) => setOutputName(e.target.value)} className="rounded border p-1" />
                  </label>
                  <div className="rounded-xl border border-[var(--line)] bg-[var(--bg-muted)] p-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">Destino en PC</div>
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <button onClick={pickMinimapSaveFile} className="rounded-full border border-[var(--line)] bg-white/70 px-4 py-2 text-xs text-[var(--text)]">Elegir destino</button>
                      <span className="text-xs text-[var(--muted)]">{minimapSaveName ? minimapSaveName : "Sin destino"}</span>
                    </div>
                    <div className="mt-2 text-xs text-[var(--muted)]">
                      {canPickMinimapSave ? "Selecciona la ruta antes de renderizar." : "Navegador sin guardado directo; usa el popup de descarga."}
                    </div>
                    {minimapSaveError && (
                      <div className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
                        {minimapSaveError}
                      </div>
                    )}
                  </div>


                </div>
                <button
                  onClick={handleRender}
                  disabled={renderBusy || routePoints.length === 0 || !backendOnline || (canPickMinimapSave && !minimapSaveHandle)}
                  className="mt-4 w-full rounded-xl bg-emerald-600 py-3 text-white font-semibold disabled:opacity-50"
                >
                  {renderBusy ? "Renderizando..." : "Generar video MP4"}
                </button>
                {jobStatus && <div className="mt-2 text-xs text-[var(--muted)]">{jobStatus.message}</div>}
                <div className="mt-3 h-2 overflow-hidden rounded-full border border-[var(--line)] bg-white/70">
                  <div
                    className="h-full bg-[var(--accent)] transition-all"
                    style={{
                      width:
                        jobStatus?.total && jobStatus.total > 0 && jobStatus.progress !== undefined
                          ? `${Math.min(100, ((jobStatus.progress ?? 0) / jobStatus.total) * 100)}%`
                          : "0%",
                    }}
                  />
                </div>
                {jobStatus?.log?.length ? (
                  <div className="mt-3 space-y-1 rounded-lg bg-[var(--bg-muted)] p-3 text-[11px] text-[var(--muted)]">
                    {jobStatus.log.slice(-4).map((line, idx) => (
                      <div key={`${line}-${idx}`}>{line}</div>
                    ))}
                  </div>
                ) : null}
                <div className="mt-3 grid gap-3 text-[11px] text-[var(--muted)] sm:grid-cols-2">
                  <div>
                    <div className="flex justify-between"><span>CPU</span><span>{Math.round(systemStats?.cpu_usage ?? 0)}%</span></div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-200">
                      <div className="h-full bg-blue-500" style={{ width: `${systemStats?.cpu_usage || 0}%` }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between"><span>GPU</span><span>{Math.round(systemStats?.gpu_usage ?? 0)}%</span></div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-200">
                      <div className="h-full bg-emerald-500" style={{ width: `${systemStats?.gpu_usage || 0}%` }} />
                    </div>
                  </div>
                </div>
                <div className="mt-2 text-[11px] text-emerald-600">
                  [SISTEMA] {systemStats?.gpu_name || "GPU..."} - {systemStats?.nvenc_available ? "NVENC" : "NVENC OFF"}
                </div>
              </section>

            </div>
          </main>
        )
      ) : (
        // ==================== PATH MODE ====================
        <main className="mx-auto flex w-full max-w-[1400px] flex-1 flex-col px-8 pb-6 md:px-14 lg:px-24">
          <div className="grid h-full gap-6 xl:grid-cols-[0.75fr_1.25fr]">
            {/* Left Panel: Controls */}
            <section className="flex h-full flex-col rounded-3xl border border-[var(--line)] bg-[var(--panel)]/85 p-5 shadow-[0_18px_45px_rgba(15,23,42,0.15)]">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-[var(--text)]">Controles</h2>
                <span className="text-xs text-[var(--muted)]">
                  {pathPoints.length ? `Puntos: ${pathPoints.length}` : "Sin puntos"}
                </span>
              </div>
              <div className="mt-4 flex flex-col gap-4">
                <label className="flex flex-col gap-2 text-sm text-[var(--text)]">
                  Video de recorrido
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime,video/*"
                    onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
                    className="rounded-2xl border border-[var(--line)] bg-white/70 px-4 py-3 text-sm text-[var(--text)] shadow-sm"
                  />
                </label>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={togglePlayback}
                    disabled={!videoUrl}
                    className="rounded-full border border-[var(--line)] bg-white/70 px-4 py-2 text-sm text-[var(--text)] transition hover:border-[var(--accent)] disabled:opacity-40"
                  >
                    Play / Pausa
                  </button>
                  <button
                    onClick={startAnnotation}
                    disabled={!videoUrl}
                    className="rounded-full border border-[var(--line)] bg-white/70 px-4 py-2 text-sm text-[var(--text)] transition hover:border-[var(--accent)] disabled:opacity-40"
                  >
                    Marcar path
                  </button>
                  <button
                    onClick={clearPoints}
                    disabled={!pathPoints.length}
                    className="rounded-full border border-[var(--line)] bg-white/70 px-4 py-2 text-sm text-[var(--text)] transition hover:border-[var(--accent)] disabled:opacity-40"
                  >
                    Limpiar
                  </button>
                </div>
                {pathError && (
                  <div className="rounded-2xl border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-600">
                    {pathError}
                  </div>
                )}
                <div className="rounded-2xl border border-[var(--line)] bg-white/70 p-4">
                  <p className="text-[10px] lowercase tracking-[0.2em] text-[var(--muted)]">
                    estilo de linea
                  </p>
                  <div className="mt-3 grid grid-cols-[1fr_1fr] gap-3 text-xs text-[var(--text)]">
                    <label className="flex flex-col gap-2">
                      Color
                      <input
                        type="color"
                        value={pathLineColor}
                        onChange={(e) => setPathLineColor(e.target.value)}
                        className="h-10 w-full rounded-xl border border-[var(--line)] bg-white"
                      />
                    </label>
                    <label className="flex flex-col gap-2">
                      Grosor ({pathLineWidth}px)
                      <input
                        type="range"
                        min={2}
                        max={12}
                        value={pathLineWidth}
                        onChange={(e) => setPathLineWidth(Number(e.target.value))}
                        className="accent-[var(--accent)]"
                      />
                    </label>
                  </div>
                  <div className="mt-3 flex h-4 w-full items-center rounded-full border border-[var(--line)] bg-white/80 px-2">
                    <div
                      className="w-full rounded-full"
                      style={{
                        background: pathLineColor,
                        height: Math.max(2, pathLineWidth),
                      }}
                    />
                  </div>
                </div>

                <button
                  onClick={runTracking}
                  disabled={!videoFile || pathPoints.length === 0 || !backendOnline}
                  className="rounded-full border border-[var(--accent)] px-6 py-2 text-sm font-semibold text-[var(--accent)] transition hover:bg-[rgba(58,167,255,0.15)] disabled:opacity-40"
                >
                  Generar path
                </button>

                {trackPath && (
                  <div className="grid gap-3">
                    <label className="text-xs text-[var(--muted)]">
                      Selecciona donde guardar el MP4
                    </label>
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        onClick={pickSaveFile}
                        className="rounded-full border border-[var(--line)] bg-white/70 px-4 py-2 text-sm text-[var(--text)] shadow-sm transition hover:border-[var(--accent)]"
                      >
                        Elegir destino
                      </button>
                      <span className="text-xs text-[var(--muted)]">
                        {saveFileName ? saveFileName : "Sin archivo seleccionado"}
                      </span>
                    </div>
                    <button
                      onClick={renderOverlay}
                      disabled={!trackPath || !saveHandle}
                      className="rounded-full bg-[var(--accent)] px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:brightness-105 disabled:opacity-50"
                    >
                      Generar video con path
                    </button>
                  </div>
                )}

                <div className="rounded-2xl border border-[var(--line)] bg-white/70 p-3 text-xs text-[var(--muted)]">
                  {pathJobStatus?.message ?? "Sin tracking todavia."}
                  <div className="mt-3 h-2 overflow-hidden rounded-full border border-[var(--line)] bg-white/70">
                    <div
                      className="h-full bg-[var(--accent)] transition-all"
                      style={{
                        width:
                          pathJobStatus?.total && pathJobStatus.total > 0 && pathJobStatus.progress !== undefined
                            ? `${Math.min(100, ((pathJobStatus.progress ?? 0) / pathJobStatus.total) * 100)}%`
                            : "0%",
                      }}
                    />
                  </div>
                  {pathJobStatus?.status === "tracking" && pathJobStatus.log?.length ? (
                    <div className="mt-3 space-y-1 text-[11px] text-[var(--muted)]">
                      {pathJobStatus.log.slice(-3).map((entry: string, idx: number) => (
                        <div key={`${entry}-${idx}`}>{entry}</div>
                      ))}
                    </div>
                  ) : null}
                </div>
                <div className="rounded-2xl border border-[var(--line)] bg-white/70 p-3 text-xs text-[var(--muted)]">
                  {overlayJobStatus?.message ?? "Sin render de overlay todavia."}
                </div>
                <div className="h-2 overflow-hidden rounded-full border border-[var(--line)] bg-white/70">
                  <div
                    className="h-full bg-[var(--accent)] transition-all"
                    style={{
                      width:
                        overlayJobStatus?.total && overlayJobStatus.total > 0 && overlayJobStatus.progress !== undefined
                          ? `${Math.min(100, ((overlayJobStatus.progress ?? 0) / overlayJobStatus.total) * 100)}%`
                          : "0%",
                    }}
                  />
                </div>
                {exportPath && (
                  <a
                    href={`${API}/download?path=${encodeURIComponent(exportPath)}`}
                    className="inline-flex items-center justify-center rounded-full border border-[var(--accent)] px-5 py-2 text-sm font-semibold text-[var(--accent)] transition hover:bg-[rgba(58,167,255,0.12)]"
                  >
                    Descargar overlay MP4
                  </a>
                )}
              </div>
            </section>

            {/* Right Panel: Video Player */}
            <section className="flex h-full flex-col rounded-3xl border border-[var(--line)] bg-[var(--panel)]/85 p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-[var(--text)]">
                  {exportPath
                    ? "Resultado final"
                    : trackData
                      ? "Vista previa"
                      : "Video original"}
                </h2>
                <span className="text-xs text-[var(--muted)]">
                  {videoUrl ? `${videoAspect.toFixed(2)} : 1` : "Sin video"}
                </span>
              </div>
              <div
                ref={playerWrapRef}
                className="mt-4 flex min-h-0 flex-1 items-center justify-center rounded-2xl border border-[var(--line)] bg-white/70 p-3"
              >
                {exportPath ? (
                  <video
                    src={`${API}/file?path=${encodeURIComponent(exportPath)}`}
                    controls
                    className="h-full w-full rounded-2xl object-contain shadow-xl"
                  />
                ) : videoUrl ? (
                  <div
                    className="relative"
                    style={{
                      width: playerWidth ? `${playerWidth}px` : "100%",
                      height: playerWidth ? `${playerWidth / videoAspect}px` : "auto",
                      maxWidth: "100%",
                    }}
                  >
                    <video
                      ref={videoElement}
                      src={videoUrl}
                      controls
                      onLoadedMetadata={() => {
                        const video = videoElement.current;
                        if (!video) return;
                        if (video.videoWidth && video.videoHeight) {
                          setVideoAspect(video.videoWidth / video.videoHeight);
                        }
                      }}
                      className={`h-full w-full rounded-2xl object-contain shadow-xl ${annotating ? "pointer-events-none" : ""}`}
                      playsInline
                    />
                    {trackPath && !trackData && (
                      <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-white/70 text-sm text-[var(--muted)]">
                        Generando vista previa...
                      </div>
                    )}
                    <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-white/80 px-3 py-1 text-[10px] text-[var(--muted)] shadow">
                      <span>Linea</span>
                      <span
                        className="block rounded-full"
                        style={{
                          width: 52,
                          height: Math.max(2, pathLineWidth),
                          background: pathLineColor,
                        }}
                      />
                    </div>
                    <div
                      className={`absolute inset-0 ${annotating ? "cursor-crosshair" : ""}`}
                      style={{ pointerEvents: annotating ? "auto" : "none" }}
                      onClick={handleVideoClick}
                    />
                    {trackData && (
                      <canvas ref={overlayCanvas} className="absolute inset-0" style={{ pointerEvents: "none" }} />
                    )}
                  </div>
                ) : (
                  <div className="flex h-full w-full items-center justify-center text-sm text-[var(--muted)]">
                    Carga un video para empezar.
                  </div>
                )}
              </div>
              <p className="mt-3 text-xs text-[var(--muted)]">
                Haz clic para marcar puntos; los puntos se muestran unos segundos y luego se
                desvanecen.
              </p>
            </section>
          </div>
        </main>
      )
      }

      {/* Video Modal (Minimap) */}
      {
        showVideoModal && minimapExportPath && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
            <div className="w-full max-w-4xl rounded-2xl bg-white p-6">
              <video src={`${API}/file?path=${encodeURIComponent(minimapExportPath)}`} controls autoPlay className="w-full rounded-lg" />
              <div className="mt-4 flex flex-wrap items-center gap-3">
                {canPickMinimapSave ? (
                  <button
                    onClick={() => (minimapSaveHandle ? saveMinimapToDisk(minimapExportPath) : pickMinimapSaveFile())}
                    className="rounded-lg bg-[var(--accent)] px-5 py-2 text-sm font-semibold text-white"
                  >
                    {minimapSaving ? "Guardando..." : minimapSaveHandle ? "Guardar en PC" : "Elegir destino"}
                  </button>
                ) : (
                  <a
                    href={`${API}/download?path=${encodeURIComponent(minimapExportPath)}`}
                    className="rounded-lg bg-[var(--accent)] px-5 py-2 text-sm font-semibold text-white"
                  >
                    Descargar MP4
                  </a>
                )}
                <button onClick={() => setShowVideoModal(false)} className="rounded-lg bg-slate-200 px-5 py-2 text-sm">
                  Cerrar
                </button>
              </div>
              {minimapSavedPath === minimapExportPath && (
                <div className="mt-2 text-xs text-emerald-600">Video guardado en tu PC.</div>
              )}
              {minimapSaveError && (
                <div className="mt-2 text-xs text-red-600">{minimapSaveError}</div>
              )}
            </div>
          </div>
        )
      }
    </div >
  );
}
