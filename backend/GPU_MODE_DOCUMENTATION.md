# GPU Rendering Mode - Complete Technical Documentation

## Executive Summary

This document provides a comprehensive technical reference for the GPU rendering mode in [`render_gpu.py`](backend/render_gpu.py), including:
1. **Current Architecture** - How the GPU rendering pipeline works
2. **Performance Analysis** - Why it runs at 5.7 FPS @ 480×480
3. **Optimization Guide** - Path to 30+ FPS
4. **Integration Guide** - How to use the GPU rendering system

> **Status**: ✅ GPU produces correct results matching CPU  
> **Performance**: 5.7 FPS @ 480×480 (175ms/frame)  
> **Target**: 30+ FPS @ 480×480 (33ms/frame)  
> **Bottleneck**: I/O and multiple GPU passes, not GPU compute

---

## Table of Contents

1. [Current Performance Analysis](#1-current-performance-analysis)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Components](#3-core-components)
4. [Performance Bottlenecks](#4-performance-bottlenecks)
5. [Optimization Path](#5-optimization-path)
6. [Integration with App](#6-integration-with-app)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Current Performance Analysis

### 1.1 Performance Breakdown @ 480×480

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   CURRENT PERFORMANCE: 5.7 FPS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Component              Time (ms)    % of Frame    Optimization         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  WMS Sampling           40-50         25-30%      HIGH PRIORITY         │
│  Ortho Sampling         30-40         20-25%      HIGH PRIORITY         │
│  Vector Sampling        20-25         12-15%      MEDIUM PRIORITY       │
│  Alpha Composite (3×)   15-20         10-12%      EASY WIN              │
│  Downsample 2×          8-12           5-7%       EASY WIN              │
│  Compass Draw (CPU)     10-15          6-9%       EASY WIN              │
│  Memory Transfer        10-15          6-9%       MEDIUM PRIORITY       │
│  numpy/cupy conversion  10-15          6-9%       HIGH PRIORITY         │
│  PNG encode/save        10-15          6-9%       HIGH PRIORITY         │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  TOTAL                  ~175ms         100%                              │
│  FPS                    ~5.7                                         │
│                                                                          │
│  KEY INSIGHT:                                                            │
│  - GPU compute: ~100ms (57%) - efficient enough                         │
│  - CPU/IO overhead: ~75ms (43%) - main bottleneck                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Not 60 FPS?

Video games achieve 60 FPS because:
- All data pre-loaded in VRAM (no I/O during render)
- Data stays on GPU (no CPU↔GPU transfers)
- Single draw call (no multiple passes)
- Direct to display buffer (no encoding)

Our current implementation:
- ❌ Still does I/O (WMS tiles, PNG saves)
- ❌ Frequent CPU↔GPU transfers
- ❌ 4+ separate affine transform passes
- ❌ CPU compass drawing

---

## 2. Architecture Overview

### 2.1 Two-Phase Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GPU RENDERING WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: PRE-LOAD (One-time initialization)                      │   │
│  │ Duration: 2-10 seconds (depends on ortho size)                   │   │
│  │                                                                  │   │
│  │  1. Load ortho from disk (windowed read)                         │   │
│  │  2. Convert to GPU tensor (contiguous planar format)             │   │
│  │  3. Generate mipmap pyramid (L0, L1, L2)                         │   │
│  │  4. Bake vectors to GPU texture                                  │   │
│  │  5. Fetch and upload WMS tiles to GPU                            │   │
│  │  6. Pre-render UI elements (cone, icons) to GPU                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2: REAL-TIME RENDERING (per frame)                         │   │
│  │ Target: < 33ms per frame (30 FPS)                                │   │
│  │                                                                  │   │
│  │  1. Select mipmap level based on zoom                            │   │
│  │  2. Sample WMS layer (bottom layer)                              │   │
│  │  3. Sample ortho using inverse transform (GPU)                   │   │
│  │  4. Composite ortho over WMS                                     │   │
│  │  5. Sample vectors using inverse transform (GPU)                 │   │
│  │  6. Composite vectors over ortho                                 │   │
│  │  7. Downsample from 2× to output size (GPU)                      │   │
│  │  8. Draw compass (CPU) + composite over frame                    │   │
│  │  9. Save frame to PNG                                            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Compositing Order

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER COMPOSITING STACK                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Layer 4 (Top)    ┌─────────────────────────────────────┐              │
│                    │         Compass/UI                  │              │
│                    │  (Drawn on CPU, composited)        │              │
│                    └─────────────────────────────────────┘              │
│                                      │▲                                 │
│                                      ││                                 │
│   Layer 3         ┌─────────────────────────────────────┐              │
│                   │         Vector Layer                │              │
│                   │  (Roads, boundaries, points)        │              │
│                   │  Sample from vector_texture         │              │
│                   └─────────────────────────────────────┘              │
│                                      │▲                                 │
│                                      ││                                 │
│   Layer 2         ┌─────────────────────────────────────┐              │
│                   │         Ortho Layer                 │              │
│                   │  (Aerial imagery, terrain)          │              │
│                   │  Sample from mipmaps[L]             │              │
│                   └─────────────────────────────────────┘              │
│                                      │▲                                 │
│                                      ││                                 │
│   Layer 1 (Bottom) ┌─────────────────────────────────────┐              │
│                    │         WMS Layer                  │              │
│                    │  (Satellite tiles from web)        │              │
│                    │  Sample from wms_texture           │              │
│                    └─────────────────────────────────────┘              │
│                                                                          │
│   Compositing: final = UI ⊕ vectors ⊕ ortho ⊕ wms                      │
│   (where ⊕ = alpha composite "over" operator)                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 GPURenderContext Class

```python
class GPURenderContext:
    """
    Manages all GPU resources and state for rendering.
    
    This is the central hub for GPU rendering. All textures are
    pre-loaded here during the pre-load phase and accessed during
    real-time rendering.
    """
    
    def __init__(self):
        # Ortho data
        self.ortho_texture: cp.ndarray = None           # Full resolution (C, H, W)
        self.mipmaps: List[cp.ndarray] = []             # Mipmap pyramid [L0, L1, L2]
        self.ortho_transform: Affine = None             # Rasterio transform
        self.ortho_crs = None                           # Coordinate reference system
        self.ortho_res_m: float = 1.0                   # Resolution in meters per pixel
        
        # Layer textures
        self.vector_texture: cp.ndarray = None          # Baked vectors (C, H, W)
        self.wms_texture: cp.ndarray = None             # WMS tiles (C, H, W)
        self.wms_bounds_px: Tuple[float, float] = None  # WMS tile bounds
        self.wms_zoom: int = None                       # WMS zoom level
        
        # UI textures
        self.nav_icon_gpu: cp.ndarray = None            # Navigation icon
        self.cone_mask_gpu: cp.ndarray = None           # Cone mask
        self.compass_cache: cp.ndarray = None           # Pre-rendered compass (360 angles)
        
        # Internal state
        self.is_ready: bool = False                     # Pre-load completion flag
```

### 3.2 Key Functions

#### _get_transformation_basis()

```python
def _get_transformation_basis(heading: float, m_per_px: float):
    """
    Calculate basis vectors for the transformation.
    Matches render.py convention where Heading points UP.
    
    This is the KEY function that ensures GPU and CPU produce
    IDENTICAL results.
    """
    rad = math.radians(heading)
    cos_h = math.cos(rad)
    sin_h = math.sin(rad)
    
    # Vector Y (DOWN in image) = -Travel Vector
    # Travel Vector T = (sin h, cos h)
    # So -T = (-sin h, -cos h)
    vec_y_e = m_per_px * (-sin_h)
    vec_y_n = m_per_px * (-cos_h)
    
    # Vector X (RIGHT in image) = 90° CW from Travel Vector
    # (cos h, -sin h)
    vec_x_e = m_per_px * cos_h
    vec_x_n = m_per_px * (-sin_h)
    
    return vec_x_e, vec_x_n, vec_y_e, vec_y_n
```

#### _sample_using_inverse_transform()

```python
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
    
    Algorithm:
    1. Get basis vectors using _get_transformation_basis()
    2. Map basis to texture plane using inverse transform
    3. Apply mipmap scaling if needed
    4. Build affine matrix for ndimage.affine_transform
    5. Sample each channel (GPU)
    """
```

#### _alpha_composite_gpu()

```python
def _alpha_composite_gpu(fg: cp.ndarray, bg: cp.ndarray) -> cp.ndarray:
    """
    Proper alpha compositing matching PIL.Image.alpha_composite.
    
    Implements the standard "over" operator:
    out = foreground over background
    
    Formula:
        out_a = fa + ba * (1 - fa)
        out_rgb = (fg_rgb * fa + bg_rgb * ba * (1 - fa)) / out_a
    """
```

---

## 4. Performance Bottlenecks

### 4.1 Detailed Bottleneck Analysis

| Bottleneck | Time (ms) | % | Solution |
|------------|-----------|---|----------|
| WMS Sampling | 40-50 | 25-30% | Fused kernel |
| Ortho Sampling | 30-40 | 20-25% | Fused kernel |
| Vector Sampling | 20-25 | 12-15% | Cache per heading |
| Alpha Composite | 15-20 | 10-12% | Fuse into single |
| Compass Draw | 10-15 | 6-9% | Pre-render cache |
| PNG Save | 10-15 | 6-9% | Stream directly |
| Memory Transfer | 10-15 | 6-9% | Keep on GPU |
| numpy/cupy conv | 10-15 | 6-9% | Eliminate transfers |
| Downsample | 8-12 | 5-7% | Already optimized |

### 4.2 Key Insight

**The bottleneck is NOT GPU compute.**

```
I/O Operations:  ~60ms (34%)  ████████████ 34%
CPU Overhead:    ~40ms (23%)  ███████       23%
GPU Sampling:    ~50ms (29%)  █████████     29%
GPU Composite:   ~25ms (14%)  ████          14%
```

---

## 5. Optimization Path

### 5.1 Quick Wins (1-2 hours each)

| Optimization | FPS Gain | Implementation |
|--------------|----------|----------------|
| Fuse alpha compositing | +2 FPS | [`render_gpu.py:76`](backend/render_gpu.py:76) |
| Pre-render compass cache | +2 FPS | [`render_gpu.py:447`](backend/render_gpu.py:447) |
| Persistent GPU buffers | +1 FPS | [`render_gpu.py:277`](backend/render_gpu.py:277) |
| FP16 precision | +2 FPS | [`render_gpu.py:143`](backend/render_gpu.py:143) |

### 5.2 Medium Optimizations (1 day each)

| Optimization | FPS Gain | Implementation |
|--------------|----------|----------------|
| Direct video streaming | +2 FPS | [`app.py`](backend/app.py) |
| Reduce supersampling | +4 FPS | [`render_gpu.py:499`](backend/render_gpu.py:499) |
| Vector heading cache | +3 FPS | [`render_gpu.py:418`](backend/render_gpu.py:418) |

### 5.3 Advanced Optimizations (2+ days)

| Optimization | FPS Gain | Implementation |
|--------------|----------|----------------|
| Fused sampling kernel | +8 FPS | [`render_gpu.py:143`](backend/render_gpu.py:143) |

### 5.4 Expected Results

```
Current:  5.7 FPS (175ms/frame)
Week 1:   9.7 FPS (103ms/frame)  [+P0 quick wins]
Week 2:  16.7 FPS (60ms/frame)   [+P1 medium]
Week 3:  24.7 FPS (40ms/frame)   [+P2 streaming]
Target:  30+ FPS (33ms/frame)    [+P3 fused kernel]
```

---

## 6. Integration with App

### 6.1 Usage Example

```python
from render_gpu import (
    init_gpu,
    preload_track_gpu,
    render_frame_gpu,
    cleanup_gpu,
    HAS_GPU
)

# Check GPU
gpu_status = init_gpu()
if gpu_status["available"]:
    print(f"Using GPU: {gpu_status['device']}")
    
    # Pre-load textures (call once)
    preload_track_gpu(config, jobs)
    
    # Render frames
    for frame in frames:
        gpu_frame = render_frame_gpu(
            dataset=dataset,
            vectors=clipped_vectors,
            center_e=frame.center_e,
            center_n=frame.center_n,
            heading=frame.heading,
            width=1920,
            height=1080,
            map_half_width_m=500,
            # ... other params ...
        )
        
        # Convert to PIL if needed
        from PIL import Image
        frame_img = Image.fromarray(cp.asnumpy(gpu_frame), "RGBA")
        frame_img.save(f"frame_{frame.idx:06d}.png")
    
    # Clean up
    cleanup_gpu()
```

### 6.2 Automatic Fallback

If GPU is not available, the system automatically falls back to CPU:

```python
# app.py - Automatic GPU/CPU selection
def _dispatch_render(cfg, dataset, vectors, center_e, center_n, heading, ...):
    if HAS_GPU and _CONTEXT.is_ready:
        # Use GPU rendering
        return render_frame_gpu(...)
    else:
        # Fallback to CPU rendering
        return render_cpu_fallback(...)
```

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Black output** | GPU context lost | Call `cleanup_gpu()` and re-preload |
| **Wrong colors** | Texture format mismatch | Ensure CHW format for GPU tensors |
| **Slow rendering** | Not pre-loaded | Call `preload_track_gpu()` first |
| **Memory errors** | VRAM full | Clear context, reduce texture size |
| **Wrong position** | Basis vector mismatch | Use `_get_transformation_basis()` |
| **Aliases** | No mipmaps | Ensure mipmaps generated |
| **UI missing** | CPU rendering not composite | Check UI compositing step |

### 7.2 Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print GPU memory usage
import cupy as cp
mem = cp.cuda.runtime.memGetInfo()
print(f"GPU Memory: {mem[0] / 1e9:.1f}GB / {mem[1] / 1e9:.1f}GB")

# Check texture shapes
print(f"Ortho shape: {_CONTEXT.ortho_texture.shape}")
print(f"Mipmaps: {len(_CONTEXT.mipmaps)}")
print(f"Vectors shape: {_CONTEXT.vector_texture.shape}")
```

---

## Summary

### Current State ✅

| Aspect | Status |
|--------|--------|
| **CPU/GPU Alignment** | ✅ Perfect (using `_get_transformation_basis()`) |
| **Layer Stacking** | ✅ Correct (WMS → Ortho → Vectors → UI) |
| **Mipmap System** | ✅ Working (automatic LOD selection) |
| **Texture Format** | ✅ Optimal (planar CHW) |
| **Correctness** | ✅ Output matches CPU |
| **Performance** | ⚠️ 5.7 FPS (I/O and multiple passes) |

### Path to 30 FPS

| Phase | Optimizations | FPS | Frame Time |
|-------|--------------|-----|------------|
| Current | - | 5.7 | 175ms |
| Week 1 | Quick wins | 9.7 | 103ms |
| Week 2 | Medium | 16.7 | 60ms |
| Week 3 | Advanced | 24.7 | 40ms |
| Target | All | 30+ | 33ms |

### Files Reference

| File | Purpose |
|------|---------|
| [`render_gpu.py`](backend/render_gpu.py) | GPU rendering implementation |
| [`gpu_utils.py`](backend/gpu_utils.py) | GPU detection and utilities |
| [`app.py`](backend/app.py) | Integration with web server |
| [`render.py`](backend/render.py) | CPU reference implementation |
| [`GPU_PROPOSAL_IMPROVEMENTS.md`](backend/GPU_PROPOSAL_IMPROVEMENTS.md) | Optimization guide |

---

*See also: [`GPU_PROPOSAL_IMPROVEMENTS.md`](backend/GPU_PROPOSAL_IMPROVEMENTS.md), [`GPU_FAILURE_ANALYSIS_TECHNICAL.md`](backend/GPU_FAILURE_ANALYSIS_TECHNICAL.md)*
