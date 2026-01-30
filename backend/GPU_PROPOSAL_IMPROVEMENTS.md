# GPU Rendering: Fine-Tuning Optimizations v2.0

## Executive Summary

Este documento proporciona **optimizaciones específicas e implementables** para mejorar el rendimiento del renderizado GPU **sin romper** la lógica CPU ni el frontend existente. Todas las propuestas son **ajustes finos** basados en el análisis del código actual (`render_gpu.py`).

> **Estado Actual**: Pipeline GPU funcional con precarga, muestreo y composición  
> **Objetivo**: Aumentar FPS y eliminar cuellos de botella identificados  
> **Estrategia**: Micro-optimizaciones que se acumulan para mayor velocidad

---

## Table of Contents

1. [Análisis del Código Actual](#1-análisis-del-código-actual)
2. [Cuellos de Botella Identificados](#2-cuellos-de-botella-identificados)
3. [Optimizaciones de Prioridad Alta (P0)](#3-optimizaciones-de-prioridad-alta-p0)
4. [Optimizaciones de Prioridad Media (P1)](#4-optimizaciones-de-prioridad-media-p1)
5. [Optimizaciones de Prioridad Baja (P2)](#5-optimizaciones-de-prioridad-baja-p2)
6. [Orden de Implementación Recomendado](#6-orden-de-implementación-recomendado)
7. [Notas de Seguridad](#7-notas-de-seguridad)

---

## 1. Análisis del Código Actual

### 1.1 Arquitectura del Pipeline GPU Actual

El sistema actual en `render_gpu.py` sigue este flujo:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE GPU ACTUAL (render_gpu.py)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                             │
│  │   PRELOAD   │ ◄── GPURenderContext.preload()                              │
│  │             │                                                             │
│  │  1. Leer Ortofoto (rasterio)                                              │
│  │  2. Convertir a RGBA (_to_rgba, _normalize_rgba)                          │
│  │  3. Subir a GPU: cp.asarray().transpose(2,0,1) ◄─ [CUELLO DE BOTELLA 1]  │
│  │  4. Generar Mipmaps (3 niveles)                                           │
│  │  5. Procesar Vectores (PIL/CPU → GPU)           ◄─ [CUELLO DE BOTELLA 2] │
│  │  6. Descargar WMS (_fetch_wms_mosaic_for_bounds)                          │
│  │  7. Subir WMS a GPU                                                       │
│  └──────┬──────┘                                                             │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐                                                             │
│  │ RENDER FRAME│ ◄── render_frame_gpu() - POR CADA FRAME                     │
│  │             │                                                             │
│  │  1. Calcular nivel mipmap (scale_ratio)                                   │
│  │  2. Muestrear WMS (_sample_wms_layer_gpu_approx)  ◄─ [CUELLO BOTELLA 3]  │
│  │     └─ 4 affine_transform (uno por canal RGBA)                            │
│  │  3. Muestrear Ortofoto (_sample_using_inverse_transform)                  │
│  │     └─ 4 affine_transform                        ◄─ [CUELLO BOTELLA 4]   │
│  │  4. Alpha Composite (ortho sobre WMS)                                     │
│  │  5. Muestrear Vectores (_sample_using_inverse_transform)                  │
│  │     └─ 4 affine_transform                        ◄─ [CUELLO BOTELLA 5]   │
│  │  6. Alpha Composite (vectors sobre resultado)                             │
│  │  7. Downsample 2× (_gpu_downsample_box)                                   │
│  │  8. UI Overlay (PIL/CPU)                         ◄─ [CUELLO BOTELLA 6]   │
│  │  9. Subir UI a GPU, composite final                                       │
│  └─────────────┘                                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Puntos Fuertes del Código Actual

| Aspecto | Estado | Descripción |
|---------|--------|-------------|
| **Precarga a GPU** | ✅ Implementado | Ortofoto, WMS y vectores se cargan una sola vez |
| **Mipmaps** | ✅ Implementado | 3 niveles (1×, 0.5×, 0.25×) para diferentes zoom |
| **Alpha Compositing GPU** | ✅ Implementado | `_alpha_composite_gpu()` funciona correctamente |
| **Downsample GPU** | ✅ Implementado | `_gpu_downsample_box()` eficiente para 2× |
| **Pipeline FFmpeg Directo** | ✅ Implementado | En `app.py` con memoria pinned |
| **Selección de Mipmap** | ✅ Implementado | Basado en `scale_ratio` |

### 1.3 Funciones Clave Analizadas

```python
# render_gpu.py - Líneas clave

# Línea 76-102: _alpha_composite_gpu() 
#   - Correcta implementación de Porter-Duff
#   - Convierte a float32 para precisión

# Línea 104-120: _gpu_downsample_box()
#   - Eficiente para scale=2 (box filter manual)
#   - Fallback a zoom() para otros scales

# Línea 122-141: _get_transformation_basis()
#   - Cálculo de vectores base para rotación
#   - Matches con render.py (heading apunta UP)

# Línea 143-200: _sample_using_inverse_transform()
#   - Loop sobre 4 canales RGBA (líneas 187-198) ◄─ OPTIMIZABLE
#   - Usa ndimage.affine_transform

# Línea 202-272: _sample_wms_layer_gpu_approx()
#   - Reproyección WMS → espacio de salida
#   - Loop sobre 4 canales (líneas 265-270) ◄─ OPTIMIZABLE

# Línea 382-448: render_frame_gpu()
#   - Supersampling fijo 2× (línea 405-406)
#   - UI overlay en CPU (líneas 437-446) ◄─ OPTIMIZABLE
```

---

## 2. Cuellos de Botella Identificados

### 2.1 Resumen de Cuellos de Botella

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CUELLOS DE BOTELLA (PRIORIDAD)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALTA PRIORIDAD (P0) - Mayor impacto, bajo riesgo                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                              │
│  ❶ Loop 4 canales en affine_transform (12 kernels por frame)                │
│     Ubicación: _sample_using_inverse_transform() L187-198                   │  
│                _sample_wms_layer_gpu_approx() L265-270                      │
│     Impacto: ~30-40% del tiempo de frame                                    │
│                                                                              │
│  ❷ Allocación de buffers cada frame                                         │
│     Ubicación: cp.zeros() en líneas 186, 264, 419                           │
│     Impacto: ~5-10% por GC/fragmentación                                    │
│                                                                              │
│  ❸ UI Overlay procesado en CPU cada frame                                   │
│     Ubicación: render_frame_gpu() L437-446                                  │
│     Impacto: ~10-15ms por frame (compass + cone + icon)                     │
│                                                                              │
│  MEDIA PRIORIDAD (P1) - Buen impacto, requiere más cuidado                  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                              │
│  ❹ Múltiples alpha_composite separados (3 llamadas)                         │
│     Ubicación: render_frame_gpu() L425, L431, L448                          │
│     Impacto: ~10-15ms total                                                 │
│                                                                              │
│  ❺ Conversión Transformer por frame (WMS reprojection)                      │
│     Ubicación: _sample_wms_layer_gpu_approx() L238-239                      │
│     Impacto: Overhead de pyproj cada frame                                  │
│                                                                              │
│  BAJA PRIORIDAD (P2) - Mejoras menores                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                              │
│  ❻ Supersampling fijo 2× (no adaptativo)                                    │
│     Ubicación: render_frame_gpu() L405                                      │
│     Impacto: Podría reducirse en casos de alta velocidad                    │
│                                                                              │
│  ❼ Dtype float32 para matrices (podría ser float16)                         │
│     Ubicación: Múltiples cp.array(..., dtype=cp.float32)                    │
│     Impacto: Reducción de ancho de banda de memoria                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Análisis Detallado de Cuello de Botella ❶

**Problema**: La función `_sample_using_inverse_transform()` ejecuta `ndimage.affine_transform()` **4 veces** (una por canal RGBA):

```python
# CÓDIGO ACTUAL (render_gpu.py L187-198)
result_planar = cp.zeros((4, out_h, out_w), dtype=cp.uint8)
for i in range(4):  # ◄─ LOOP INEFICIENTE
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
```

Esto significa **12 llamadas a kernel por frame** (3 capas × 4 canales). Cada llamada tiene:
- Overhead de lanzamiento de kernel (~5-10μs)
- Sincronización implícita entre kernels
- No se aprovecha el paralelismo entre canales

---

## 3. Optimizaciones de Prioridad Alta (P0)

### 3.1 P0-A: Eliminar Loop de 4 Canales con `map_coordinates`

**Objetivo**: Muestrear los 4 canales RGBA en **una sola operación** GPU.

**Solución**: Usar `cupyx.scipy.ndimage.map_coordinates` con coordenadas pre-calculadas que aplican a todos los canales simultáneamente.

```python
# PROPUESTA: Reemplazar _sample_using_inverse_transform()

from cupyx.scipy.ndimage import map_coordinates

def _sample_using_inverse_transform_optimized(
    texture_planar: cp.ndarray,  # Shape: (4, H, W)
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
    Versión optimizada: muestrea 4 canales en una sola operación GPU.
    
    Cambios respecto al original:
    - Elimina loop for i in range(4)
    - Usa map_coordinates con coordenadas expandidas
    - Mantiene misma salida: (out_h, out_w, 4) en uint8
    """
    # 1. Calcular matriz de transformación (igual que antes)
    vxe, vxn, vye, vyn = _get_transformation_basis(heading, m_per_px_out)
    itf = ~ortho_transform
    level_scale = 1.0 / (2 ** mipmap_level)
    
    cx_tex, cy_tex = itf * (center_e, center_n)
    cx_tex *= level_scale
    cy_tex *= level_scale
    
    d_col_dx = (itf.a * vxe + itf.b * vxn) * level_scale
    d_col_dy = (itf.a * vye + itf.b * vyn) * level_scale
    d_row_dx = (itf.d * vxe + itf.e * vxn) * level_scale
    d_row_dy = (itf.d * vye + itf.e * vyn) * level_scale
    
    # 2. Generar grid de coordenadas de salida
    # Usamos meshgrid en GPU para evitar transferencias
    out_y = cp.arange(out_h, dtype=cp.float32)
    out_x = cp.arange(out_w, dtype=cp.float32)
    grid_y, grid_x = cp.meshgrid(out_y, out_x, indexing='ij')
    
    # Centrar coordenadas
    grid_y_centered = grid_y - out_h / 2.0
    grid_x_centered = grid_x - out_w / 2.0
    
    # 3. Calcular coordenadas de entrada (row, col en textura)
    # Aplicar transformación inversa
    input_row = cy_tex + d_row_dy * grid_y_centered + d_row_dx * grid_x_centered
    input_col = cx_tex + d_col_dy * grid_y_centered + d_col_dx * grid_x_centered
    
    # 4. Stack para map_coordinates: shape (2, out_h, out_w)
    coordinates = cp.stack([input_row, input_col], axis=0)
    
    # 5. Muestrear TODOS los canales en una operación
    # map_coordinates con entrada (4, H, W) produce (4, out_h, out_w)
    result_planar = map_coordinates(
        texture_planar.astype(cp.float32),  # Convertir a float para interpolación
        coordinates,
        order=1,  # Bilinear
        mode='constant',
        cval=0.0,
        prefilter=False
    )
    
    # 6. Convertir de vuelta a uint8 y transponer a (H, W, 4)
    result_planar = cp.clip(result_planar, 0, 255).astype(cp.uint8)
    return cp.transpose(result_planar, (1, 2, 0))
```

**Impacto Esperado**: 
- De 4 llamadas a kernel → 1 llamada
- Ahorro estimado: **25-35% del tiempo de muestreo**

**Riesgo**: ⚠️ MEDIO - Requiere validación visual para confirmar que el muestreo es idéntico.

**Ubicación del cambio**: `render_gpu.py` líneas 143-200

---

### 3.2 P0-B: Pre-alocar Buffers de Trabajo en GPURenderContext

**Objetivo**: Eliminar allocaciones repetidas de `cp.zeros()` en cada frame.

**Solución**: Añadir buffers persistentes al `GPURenderContext` que se reutilizan.

```python
# PROPUESTA: Modificar GPURenderContext

class GPURenderContext:
    def __init__(self):
        # ... atributos existentes ...
        
        # NUEVO: Buffers de trabajo pre-alocados
        self._work_buffers_initialized = False
        self._wms_buffer: Optional[cp.ndarray] = None      # (4, max_h, max_w)
        self._ortho_buffer: Optional[cp.ndarray] = None    # (4, max_h, max_w)
        self._vector_buffer: Optional[cp.ndarray] = None   # (4, max_h, max_w)
        self._composite_buffer: Optional[cp.ndarray] = None  # (max_h, max_w, 4)
        self._max_supersampled_size: Tuple[int, int] = (0, 0)
    
    def _ensure_work_buffers(self, ss_width: int, ss_height: int):
        """
        Asegura que los buffers de trabajo estén alocados para el tamaño dado.
        Solo re-aloca si el tamaño actual es insuficiente.
        """
        if (self._work_buffers_initialized and 
            ss_width <= self._max_supersampled_size[0] and
            ss_height <= self._max_supersampled_size[1]):
            return  # Buffers existentes son suficientes
        
        # Alocar con margen del 20% para evitar re-alocaciones frecuentes
        alloc_w = int(ss_width * 1.2)
        alloc_h = int(ss_height * 1.2)
        
        # Free existing if any
        if self._wms_buffer is not None:
            del self._wms_buffer
            del self._ortho_buffer
            del self._vector_buffer
            del self._composite_buffer
            cp.get_default_memory_pool().free_all_blocks()
        
        # Allocate new
        self._wms_buffer = cp.zeros((4, alloc_h, alloc_w), dtype=cp.uint8)
        self._ortho_buffer = cp.zeros((4, alloc_h, alloc_w), dtype=cp.uint8)
        self._vector_buffer = cp.zeros((4, alloc_h, alloc_w), dtype=cp.uint8)
        self._composite_buffer = cp.zeros((alloc_h, alloc_w, 4), dtype=cp.uint8)
        self._max_supersampled_size = (alloc_w, alloc_h)
        self._work_buffers_initialized = True
        
    def get_wms_slice(self, h: int, w: int) -> cp.ndarray:
        """Retorna un slice del buffer WMS del tamaño requerido."""
        return self._wms_buffer[:, :h, :w]
    
    def get_ortho_slice(self, h: int, w: int) -> cp.ndarray:
        """Retorna un slice del buffer ortho del tamaño requerido."""
        return self._ortho_buffer[:, :h, :w]
    
    def get_vector_slice(self, h: int, w: int) -> cp.ndarray:
        """Retorna un slice del buffer vector del tamaño requerido."""
        return self._vector_buffer[:, :h, :w]
    
    def get_composite_slice(self, h: int, w: int) -> cp.ndarray:
        """Retorna un slice del buffer composite del tamaño requerido."""
        return self._composite_buffer[:h, :w, :]
```

**Uso en render_frame_gpu()**:

```python
def render_frame_gpu(...):
    # ...
    ss_factor = 2
    sw, sh = width * ss_factor, height * ss_factor
    
    # NUEVO: Asegurar buffers y usarlos
    _CONTEXT._ensure_work_buffers(sw, sh)
    
    # En lugar de: final_gpu = cp.zeros((sh, sw, 4), dtype=cp.uint8)
    # Usar slice del buffer pre-alocado:
    final_gpu = _CONTEXT.get_composite_slice(sh, sw)
    final_gpu.fill(0)  # Limpiar en lugar de alocar
    # ...
```

**Impacto Esperado**:
- Elimina overhead de allocación en cada frame
- Reduce fragmentación de memoria GPU
- Ahorro estimado: **5-10% del tiempo total**

**Riesgo**: ⚡ BAJO - Cambio de memoria, no de lógica de renderizado.

**Ubicación del cambio**: `render_gpu.py` líneas 277-302 (GPURenderContext)

---

### 3.3 P0-C: Pre-renderizar Compass a Caché GPU (360°)

**Objetivo**: Eliminar el dibujado del compass en CPU en cada frame.

**Problema Actual** (líneas 437-446):
```python
# CÓDIGO ACTUAL - CPU cada frame
ui_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
# ...
if show_compass:
    c_sz = max(15, compass_size_px)
    c_pos = (width - (c_sz // 2 + 15), c_sz // 2 + 15)
    _draw_compass(ui_layer, c_pos, c_sz, -heading)  # ◄─ CPU CADA FRAME

ui_gpu = cp.asarray(np.array(ui_layer))  # ◄─ Transferencia CPU→GPU
```

**Solución**: Pre-renderizar 360 imágenes del compass (1 por grado) durante `preload()`.

```python
# PROPUESTA: Agregar a GPURenderContext

class GPURenderContext:
    def __init__(self):
        # ... existentes ...
        
        # NUEVO: Caché de compass pre-renderizado
        self._compass_cache: Optional[cp.ndarray] = None  # Shape: (360, size, size, 4)
        self._compass_size: int = 0
    
    def _pre_render_compass_cache(self, compass_size: int):
        """
        Pre-renderiza el compass para los 360 grados.
        Se ejecuta una sola vez durante preload().
        """
        if self._compass_cache is not None and self._compass_size == compass_size:
            return  # Ya está cacheado
        
        from render import _draw_compass
        from PIL import Image
        import numpy as np
        
        # Crear array en CPU primero (360 imágenes)
        cache_cpu = np.zeros((360, compass_size, compass_size, 4), dtype=np.uint8)
        
        for heading in range(360):
            img = Image.new("RGBA", (compass_size, compass_size), (0, 0, 0, 0))
            center = (compass_size // 2, compass_size // 2)
            _draw_compass(img, center, compass_size, -heading)
            cache_cpu[heading] = np.array(img)
        
        # Subir TODO el caché a GPU de una vez
        self._compass_cache = cp.asarray(cache_cpu)
        self._compass_size = compass_size
        
        print(f"[GPU] Compass cache: {360} rotaciones pre-renderizadas ({compass_size}x{compass_size})")
    
    def get_compass_for_heading(self, heading: float) -> cp.ndarray:
        """
        Obtiene el compass pre-renderizado para el heading dado.
        Retorna: (size, size, 4) uint8 array en GPU
        """
        heading_int = int(heading) % 360
        return self._compass_cache[heading_int]
```

**Uso en preload()**:

```python
def preload(self, dataset, center_points, margin_m, vectors=None, 
            arrow_size=100, cone_len=200, wms_source="google_hybrid", 
            icon_opacity=0.4, progress_callback=None,
            compass_size=40):  # ◄─ Nuevo parámetro
    # ... código existente ...
    
    notify(95, "Pre-renderizando compass...")  # NUEVO
    self._pre_render_compass_cache(compass_size)
    
    notify(100, "Precarga GPU completada.")
    self.is_ready = True
```

**Uso en render_frame_gpu()**:

```python
def render_frame_gpu(...):
    # ... código existente hasta UI overlay ...
    
    if show_compass and _CONTEXT._compass_cache is not None:
        # NUEVO: Obtener compass pre-renderizado de GPU
        c_sz = max(15, compass_size_px)
        compass_gpu = _CONTEXT.get_compass_for_heading(heading)
        
        # Posicionar en la esquina superior derecha
        x_offset = width - c_sz - 15
        y_offset = 15
        
        # Composite directo en GPU (sin transferencia CPU)
        compass_region = final_gpu[y_offset:y_offset+c_sz, x_offset:x_offset+c_sz, :]
        final_gpu[y_offset:y_offset+c_sz, x_offset:x_offset+c_sz, :] = \
            _alpha_composite_gpu(compass_gpu, compass_region)
    
    # ... resto del código ...
```

**Impacto Esperado**:
- Elimina `_draw_compass()` por frame (~5-10ms)
- Elimina conversión PIL→numpy→cupy (~2-3ms)
- Ahorro total estimado: **10-15ms por frame**

**Riesgo**: ⚡ BAJO - Solo cambia cómo se obtiene el compass, no el resultado visual.

**Memoria GPU Adicional**: ~360 × 40 × 40 × 4 = 2.3 MB (insignificante)

---

## 4. Optimizaciones de Prioridad Media (P1)

### 4.1 P1-A: Fusionar Alpha Compositing en Operación Única

**Problema Actual**: 3 llamadas separadas a `_alpha_composite_gpu()`:

```python
# CÓDIGO ACTUAL (render_frame_gpu)
final_gpu = _alpha_composite_gpu(ortho_layer, final_gpu)   # 1
final_gpu = _alpha_composite_gpu(vec_layer, final_gpu)     # 2
# ... después ...
return _alpha_composite_gpu(ui_gpu, final_gpu)              # 3
```

**Solución**: Crear función de composición multi-capa.

```python
# PROPUESTA: Nueva función de composición fusionada

def _alpha_composite_multi_gpu(
    layers: List[cp.ndarray],
    order: str = "back_to_front"
) -> cp.ndarray:
    """
    Compone múltiples capas RGBA en una sola operación optimizada.
    
    Args:
        layers: Lista de arrays (H, W, 4) en orden de composición
        order: "back_to_front" (primero es fondo) o "front_to_back"
    
    Returns:
        Array resultante (H, W, 4) en uint8
    
    Optimización: Evita crear arrays intermedios innecesarios.
    """
    if len(layers) == 0:
        raise ValueError("Se requiere al menos una capa")
    if len(layers) == 1:
        return layers[0].copy()
    
    # Convertir todas a float32 de una vez
    alphas = [layer[:, :, 3:4].astype(cp.float32) / 255.0 for layer in layers]
    rgbs = [layer[:, :, :3].astype(cp.float32) for layer in layers]
    
    # Empezar desde el fondo
    if order == "back_to_front":
        result_rgb = rgbs[0].copy()
        result_a = alphas[0].copy()
        
        for i in range(1, len(layers)):
            fg_a = alphas[i]
            fg_rgb = rgbs[i]
            
            # Porter-Duff "over"
            inv_fg_a = 1.0 - fg_a
            out_a = fg_a + result_a * inv_fg_a
            out_a_safe = cp.where(out_a > 1e-6, out_a, 1.0)
            
            result_rgb = (fg_rgb * fg_a + result_rgb * result_a * inv_fg_a) / out_a_safe
            result_a = out_a
    else:
        raise NotImplementedError("front_to_back no implementado")
    
    # Convertir de vuelta a uint8
    return cp.dstack((
        cp.clip(result_rgb, 0, 255).astype(cp.uint8),
        cp.clip(result_a * 255, 0, 255).astype(cp.uint8)
    ))
```

**Uso en render_frame_gpu()**:

```python
# ANTES:
# final_gpu = _alpha_composite_gpu(ortho_layer, final_gpu)
# final_gpu = _alpha_composite_gpu(vec_layer, final_gpu) 
# return _alpha_composite_gpu(ui_gpu, final_gpu)

# DESPUÉS:
layers = [wms_layer, ortho_layer, vec_layer, ui_layer_gpu]
return _alpha_composite_multi_gpu(layers, order="back_to_front")
```

**Impacto Esperado**:
- Reduce conversiones float32↔uint8 intermedias
- Menos arrays temporales = menos presión de memoria
- Ahorro estimado: **5-8ms por frame**

**Riesgo**: ⚠️ MEDIO - Requiere validar que el orden de composición es correcto.

---

### 4.2 P1-B: Cachear Transformer de Proyección WMS

**Problema Actual**: Se crea un `Transformer` nuevo en cada frame:

```python
# CÓDIGO ACTUAL (_sample_wms_layer_gpu_approx L238-239)
from_crs = CRS.from_user_input(ortho_crs)
transformer = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
```

**Solución**: Cachear el transformer en `GPURenderContext`.

```python
# PROPUESTA: Modificar GPURenderContext

class GPURenderContext:
    def __init__(self):
        # ... existentes ...
        
        # NUEVO: Transformer cacheado
        self._wms_transformer: Optional[Transformer] = None
        self._wms_from_crs_str: Optional[str] = None
    
    def get_wms_transformer(self, ortho_crs) -> Transformer:
        """
        Obtiene el transformer para WMS, cacheándolo si es posible.
        """
        from pyproj import CRS, Transformer
        
        crs_str = str(ortho_crs)
        if self._wms_transformer is not None and self._wms_from_crs_str == crs_str:
            return self._wms_transformer
        
        from_crs = CRS.from_user_input(ortho_crs)
        self._wms_transformer = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
        self._wms_from_crs_str = crs_str
        return self._wms_transformer
```

**Uso en _sample_wms_layer_gpu_approx()**:

```python
def _sample_wms_layer_gpu_approx(..., context: GPURenderContext = None):
    # ...
    
    # ANTES:
    # from_crs = CRS.from_user_input(ortho_crs)
    # transformer = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
    
    # DESPUÉS:
    transformer = _CONTEXT.get_wms_transformer(ortho_crs)
    
    # ... resto igual ...
```

**Impacto Esperado**:
- Elimina creación de objetos pyproj cada frame
- Ahorro estimado: **2-3ms por frame**

**Riesgo**: ⚡ BAJO - Solo caché de objeto inmutable.

---

### 4.3 P1-C: Pre-calcular Matriz de Transformación Afín

**Optimización menor**: Los cálculos de `d_col_dx`, `d_row_dy`, etc. son determinísticos dado `(heading, m_per_px_out, ortho_transform)`. Se pueden cachear para headings similares.

```python
# PROPUESTA: Agregar caché de matrices afines

class GPURenderContext:
    def __init__(self):
        # ... existentes ...
        self._affine_matrix_cache: Dict[Tuple[int, int, int], Tuple[cp.ndarray, cp.ndarray]] = {}
        self._affine_cache_max_size: int = 180  # Limitar a 180 entradas
    
    def get_cached_affine(
        self, 
        heading: float, 
        m_per_px_out: float, 
        mipmap_level: int
    ) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        """
        Obtiene matriz y offset cacheados si existen.
        Discretiza heading a 2 decimales para aumentar cache hits.
        """
        # Discretizar para aumentar hits
        key = (
            round(heading, 1),  # Redondear heading a 0.1 grados
            round(m_per_px_out * 1000),  # Discretizar a mm
            mipmap_level
        )
        return self._affine_matrix_cache.get(key)
    
    def cache_affine(
        self,
        heading: float,
        m_per_px_out: float,
        mipmap_level: int,
        matrix: cp.ndarray,
        offset: cp.ndarray
    ):
        """Cachea matriz y offset para uso futuro."""
        if len(self._affine_matrix_cache) >= self._affine_cache_max_size:
            # LRU simple: eliminar primera entrada
            first_key = next(iter(self._affine_matrix_cache))
            del self._affine_matrix_cache[first_key]
        
        key = (
            round(heading, 1),
            round(m_per_px_out * 1000),
            mipmap_level
        )
        self._affine_matrix_cache[key] = (matrix.copy(), offset.copy())
```

**Impacto Esperado**: 
- Menor para tracks con headings variados
- Significativo para tracks con headings repetidos
- Ahorro estimado: **1-3ms por frame** en mejores casos

---

## 5. Optimizaciones de Prioridad Baja (P2)

### 5.1 P2-A: Supersampling Adaptativo

**Idea**: Reducir supersampling cuando el movimiento es rápido (motion blur oculta aliasing).

```python
# PROPUESTA: Modificar render_frame_gpu()

def _compute_adaptive_supersample(
    velocity: Optional[float] = None,
    frame_rate: int = 30
) -> int:
    """
    Calcula factor de supersampling óptimo.
    
    Args:
        velocity: Velocidad en m/s (None = usar default)
        frame_rate: FPS del video
    
    Returns:
        Factor de supersampling (1 o 2)
    """
    if velocity is None:
        return 2  # Default actual
    
    # A velocidades altas, el motion blur natural oculta el aliasing
    # Umbral empírico: > 15 m/s = ~54 km/h
    if velocity > 15:
        return 1
    return 2
```

**NOTA**: Esta optimización requiere que el sistema de jobs pase la velocidad al renderizador, lo cual es un cambio de interfaz. Se recomienda como optimización futura.

**Impacto Potencial**: Reducir resolución interna 4× (de 960×960 a 480×480) = **~30-40% más rápido** en frames de alta velocidad.

**Riesgo**: ⚠️ MEDIO - Puede introducir aliasing visible. Requiere pruebas extensivas.

---

### 5.2 P2-B: Usar Float16 para Muestreo

**Idea**: Reducir precision de float32 a float16 para operaciones de muestreo.

```python
# PROPUESTA: Modificar _sample_using_inverse_transform()

# En lugar de:
matrix = cp.array([...], dtype=cp.float32)

# Usar:
matrix = cp.array([...], dtype=cp.float16)  # Half precision

# NOTA: map_coordinates de CuPy acepta float16
```

**Impacto Potencial**:
- 2× menos ancho de banda de memoria
- RTX 3050 tiene buen soporte FP16
- Ahorro estimado: **5-10%** en operaciones memory-bound

**Riesgo**: ⚡ BAJO para muestreo de texturas (la precisión no afecta visualmente).

**NOTA**: Requiere pruebas para confirmar que no hay artefactos en coordenadas extremas.

---

## 6. Orden de Implementación Recomendado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PLAN DE IMPLEMENTACIÓN SUGERIDO                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1: Quick Wins (1-2 días)                                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                │
│  □ P0-B: Pre-alocar buffers de trabajo                                       │
│    → Bajo riesgo, fácil implementación                                       │
│    → Impacto: ~5-10% mejora                                                  │
│                                                                              │
│  □ P1-B: Cachear Transformer de WMS                                          │
│    → Cambio trivial, 0 riesgo                                                │
│    → Impacto: ~2-3ms por frame                                               │
│                                                                              │
│  □ P0-C: Pre-renderizar compass 360°                                         │
│    → Bajo riesgo, alto impacto                                               │
│    → Impacto: ~10-15ms por frame                                             │
│                                                                              │
│  FASE 2: Optimización Core (2-3 días)                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                           │
│  □ P0-A: Eliminar loop 4 canales con map_coordinates                         │
│    → Requiere testing visual cuidadoso                                       │
│    → ALTO impacto: ~25-35% mejora en muestreo                                │
│                                                                              │
│  □ P1-A: Fusionar alpha compositing                                          │
│    → Requiere validar orden de capas                                         │
│    → Impacto: ~5-8ms por frame                                               │
│                                                                              │
│  FASE 3: Polish (opcional)                                                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━                                                   │
│  □ P1-C: Cachear matrices afines                                             │
│  □ P2-B: Float16 para muestreo                                               │
│  □ P2-A: Supersampling adaptativo (requiere cambios de interfaz)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Estimación de Mejora Acumulativa

| Fase | Optimizaciones | FPS Estimados* | Mejora |
|------|---------------|----------------|--------|
| Actual | Baseline | ~5-6 FPS | - |
| Fase 1 | P0-B, P1-B, P0-C | ~8-10 FPS | +40-60% |
| Fase 2 | + P0-A, P1-A | ~12-15 FPS | +50-80% |
| Fase 3 | + P1-C, P2-B | ~15-18 FPS | +15-20% |

*Estimaciones basadas en análisis de código. Resultados reales dependen del hardware y contenido.

---

## 7. Notas de Seguridad

### 7.1 Qué NO Modificar

| Archivo | Razón |
|---------|-------|
| `render.py` | Lógica CPU de fallback - INTOCABLE |
| `app.py` (endpoints) | API pública - NO cambiar firmas |
| `models.py` | Modelos de datos - NO modificar |
| `web/` (frontend) | Funciona correctamente - NO tocar |

### 7.2 Validación OBLIGATORIA: 3 Tests Después de Cada Cambio

> ⚠️ **CRÍTICO**: Después de implementar CUALQUIER optimización de este documento, se DEBEN ejecutar los siguientes 3 tests de validación usando los datos de prueba ubicados en:
> 
> **`D:\Dev\MinimapaGPU\backend\gpu_validation\`**

#### Archivos de Validación Disponibles

| Archivo | Descripción | Uso |
|---------|-------------|-----|
| `test_ortho_crop.tif` | Ortomosaico de prueba (15.5 MB) | Input principal para tests |
| `LinderoGeneral.geojson` | Vectores de linderos (137 KB) | Test de renderizado de vectores |
| `Vias.geojson` | Vectores de vías (200 KB) | Test de renderizado de líneas |
| `render_cpu.png` | Referencia de renderizado CPU | Comparación visual baseline |
| `render_gpu_fixed.png` | Referencia GPU correcta | Comparación para regresiones |
| `diff.png` | Imagen de diferencias | Referencia de tolerancia aceptable |
| `pipe_test.mp4` | Video de referencia funcional | Validación de pipeline completo |

---

#### TEST 1: Validación de Frame Estático (Comparación Visual)

**Objetivo**: Verificar que el frame renderizado GPU sea visualmente idéntico al CPU.

```python
# TEST 1: Ejecutar desde backend/
# Comando sugerido (adaptar según implementación actual)

import numpy as np
from PIL import Image
from pathlib import Path

def test_static_frame_comparison():
    """
    Compara un frame GPU vs la referencia CPU.
    DEBE ejecutarse después de cada cambio.
    """
    VALIDATION_DIR = Path("gpu_validation")
    
    # 1. Renderizar frame con configuración estándar
    # (usar test_ortho_crop.tif, LinderoGeneral.geojson, Vias.geojson)
    
    # 2. Cargar referencia CPU
    ref_cpu = np.array(Image.open(VALIDATION_DIR / "render_cpu.png"))
    
    # 3. Comparar pixel a pixel
    # Tolerancia: ±3 por canal (para diferencias de interpolación)
    diff = np.abs(rendered_gpu.astype(np.int16) - ref_cpu.astype(np.int16))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # 4. Criterios de PASS/FAIL
    assert max_diff <= 10, f"FAIL: Diferencia máxima {max_diff} > 10"
    assert mean_diff <= 1.0, f"FAIL: Diferencia media {mean_diff} > 1.0"
    
    print(f"✅ TEST 1 PASSED: max_diff={max_diff}, mean_diff={mean_diff:.3f}")
```

**Criterio de Aceptación**: 
- Diferencia máxima por pixel ≤ 10 (de 255)
- Diferencia media ≤ 1.0

---

#### TEST 2: Validación de Video Completo (Pipeline E2E)

**Objetivo**: Generar un video corto y verificar integridad del pipeline.

```python
def test_video_pipeline():
    """
    Genera un video de 5 segundos (150 frames @ 30fps) y valida:
    - No hay frames corruptos
    - Video es reproducible
    - Tamaño de archivo razonable
    """
    VALIDATION_DIR = Path("gpu_validation")
    
    # 1. Generar video usando:
    #    - Ortho: test_ortho_crop.tif
    #    - Vectores: LinderoGeneral.geojson + Vias.geojson
    #    - Track: ruta circular simple (generar programáticamente)
    #    - WMS: google_hybrid
    
    # 2. Validar que el archivo existe y tiene tamaño > 100KB
    output = Path("gpu_validation/test_output.mp4")
    assert output.exists(), "FAIL: Video no generado"
    assert output.stat().st_size > 100_000, "FAIL: Video muy pequeño (corrupto?)"
    
    # 3. Validar con ffprobe que es reproducible
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", str(output)],
        capture_output=True, text=True
    )
    assert "duration=" in result.stdout, "FAIL: Video no es reproducible"
    
    print(f"✅ TEST 2 PASSED: Video generado correctamente")
```

**Criterio de Aceptación**:
- Video generado sin errores
- Archivo > 100 KB
- FFprobe puede leer duración

---

#### TEST 3: Validación de Performance (Benchmark)

**Objetivo**: Verificar que la optimización NO degrada el rendimiento.

```python
import time

def test_performance_benchmark():
    """
    Mide tiempo de renderizado de 100 frames.
    Compara contra baseline para detectar regresiones.
    """
    BASELINE_FPS = 5.0  # FPS mínimo aceptable (ajustar según baseline actual)
    NUM_FRAMES = 100
    
    # 1. Precarga
    # preload_track_gpu(...)
    
    # 2. Renderizar 100 frames y medir tiempo
    start = time.perf_counter()
    for i in range(NUM_FRAMES):
        # render_frame_gpu(...)
        pass
    elapsed = time.perf_counter() - start
    
    # 3. Calcular FPS
    fps = NUM_FRAMES / elapsed
    
    # 4. Validar que no hay regresión
    assert fps >= BASELINE_FPS * 0.9, f"FAIL: Regresión de performance! {fps:.1f} < {BASELINE_FPS * 0.9:.1f}"
    
    print(f"✅ TEST 3 PASSED: {fps:.1f} FPS (baseline: {BASELINE_FPS} FPS)")
    
    # 5. Reportar mejora si la hay
    if fps > BASELINE_FPS * 1.1:
        improvement = ((fps / BASELINE_FPS) - 1) * 100
        print(f"🚀 MEJORA DETECTADA: +{improvement:.1f}%")
```

**Criterio de Aceptación**:
- FPS ≥ 90% del baseline
- Idealmente: FPS > baseline

---

### 7.3 Criterios de Aceptación del Video Final

> ⚠️ **OBLIGATORIO**: El video generado DEBE contener TODOS los siguientes elementos visuales, tal como funciona la implementación actual.

#### Elementos OBLIGATORIOS en el Video

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              CHECKLIST DE ELEMENTOS VISUALES OBLIGATORIOS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ☑ BRÚJULA (Compass)                                                         │
│    └─ Debe aparecer en esquina superior derecha                              │
│    └─ Debe rotar correctamente según heading                                 │
│    └─ Debe tener el diseño actual (N, S, E, W visibles)                     │
│                                                                              │
│  ☑ ICONO DE NAVEGACIÓN (Nav Icon)                                           │
│    └─ Debe aparecer centrado en el frame                                     │
│    └─ Flecha/triángulo apuntando hacia arriba (dirección de viaje)          │
│                                                                              │
│  ☑ CÍRCULO OPACO DEL ICONO                                                  │
│    └─ Círculo semitransparente detrás del icono                              │
│    └─ Opacidad configurable (icon_circle_opacity)                            │
│    └─ Tamaño configurable (icon_circle_size_px)                              │
│                                                                              │
│  ☑ CONO DE VISIÓN (Vision Cone)                                             │
│    └─ Cono semitransparente apuntando hacia adelante                         │
│    └─ Ángulo y longitud configurables                                        │
│                                                                              │
│  ☑ VECTORES GEOJSON                                                         │
│    └─ Líneas de LinderoGeneral.geojson visibles                              │
│    └─ Líneas de Vias.geojson visibles                                        │
│    └─ Colores correctos según configuración                                  │
│    └─ Grosor de línea correcto                                               │
│                                                                              │
│  ☑ MAPA BASE SATELITAL (WMS Layer)                                          │
│    └─ Imagen satelital visible como fondo                                    │
│    └─ Fuente seleccionada (google_hybrid, esri, bing, etc.)                 │
│    └─ Resolución apropiada para el nivel de zoom                             │
│                                                                              │
│  ☑ ORTOMOSAICO TIFF                                                         │
│    └─ Imagen del TIFF cargado superpuesta sobre WMS                          │
│    └─ Transparencia correcta en bordes                                       │
│    └─ Colores fieles al original                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Defectos NO PERMITIDOS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEFECTOS QUE CAUSAN RECHAZO                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✖ HUECOS NEGROS (Black Holes)                                              │
│    └─ NO debe haber áreas negras donde debería verse contenido               │
│    └─ Indica: fallo en muestreo o coordenadas fuera de rango                │
│                                                                              │
│  ✖ JITTER / TEMBLOR (Temporal Instability)                                  │
│    └─ NO debe haber temblor o vibración entre frames consecutivos            │
│    └─ Indica: inconsistencia en cálculos de transformación                  │
│    └─ El movimiento debe ser suave y continuo                                │
│                                                                              │
│  ✖ COSTURAS VISIBLES (Seams/Stitching Artifacts)                            │
│    └─ NO debe haber líneas o bordes visibles entre capas                     │
│    └─ Ortho y WMS deben fusionarse sin costuras                              │
│    └─ Indica: error en alineación de coordenadas                            │
│                                                                              │
│  ✖ ZOOM INCORRECTO ENTRE CAPAS                                              │
│    └─ Ortho y WMS deben tener el MISMO nivel de zoom visual                  │
│    └─ NO debe verse una capa "más cerca" que otra                            │
│    └─ Indica: error en cálculo de m_per_px o escala                         │
│                                                                              │
│  ✖ DESALINEACIÓN ESPACIAL                                                   │
│    └─ Las capas deben estar perfectamente alineadas geográficamente          │
│    └─ Los vectores deben coincidir con los features del TIFF                 │
│    └─ Indica: error en transformación de coordenadas                        │
│                                                                              │
│  ✖ ELEMENTOS UI FALTANTES                                                   │
│    └─ Brújula, icono, cono, círculo DEBEN estar presentes                    │
│    └─ Indica: regresión en código de UI overlay                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Comparación Visual de Referencia

Para validar visualmente, comparar contra:

| Referencia | Ruta | Descripción |
|------------|------|-------------|
| **CPU Baseline** | `gpu_validation/render_cpu.png` | Frame correcto renderizado por CPU |
| **GPU Correcto** | `gpu_validation/render_gpu_fixed.png` | Frame GPU que pasa validación |
| **Diff Aceptable** | `gpu_validation/diff.png` | Diferencias tolerables entre CPU/GPU |
| **Video Funcional** | `gpu_validation/pipe_test.mp4` | Video completo que funciona correctamente |

---

### 7.4 Rollback Plan

- **Git**: Cada optimización debe ser un commit separado
- **Feature flags**: Considerar `USE_OPTIMIZED_SAMPLING = True/False`
- **Logging**: Agregar logs de performance para comparar antes/después
- **Backup**: Antes de cada cambio, asegurar que el commit anterior genera video correcto

---

### 7.5 Flujo de Trabajo Obligatorio

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                FLUJO DE TRABAJO PARA CADA OPTIMIZACIÓN                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ANTES de modificar código:                                               │
│     □ Ejecutar TEST 1, 2, 3 con código ACTUAL                                │
│     □ Guardar resultados como BASELINE                                       │
│     □ Hacer commit del estado actual (punto de rollback)                     │
│                                                                              │
│  2. IMPLEMENTAR la optimización:                                             │
│     □ Seguir propuesta del documento                                         │
│     □ Documentar cualquier desviación                                        │
│                                                                              │
│  3. DESPUÉS de modificar código:                                             │
│     □ Ejecutar TEST 1: Comparación visual → DEBE PASAR                       │
│     □ Ejecutar TEST 2: Pipeline E2E → DEBE PASAR                             │
│     □ Ejecutar TEST 3: Performance → DEBE PASAR (sin regresión)              │
│                                                                              │
│  4. VALIDACIÓN VISUAL del video:                                             │
│     □ Reproducir video generado                                              │
│     □ Verificar checklist de elementos obligatorios                          │
│     □ Verificar ausencia de defectos prohibidos                              │
│                                                                              │
│  5. Si TODO pasa:                                                            │
│     □ Hacer commit con mensaje descriptivo                                   │
│     □ Actualizar baseline de performance si mejoró                           │
│                                                                              │
│  6. Si ALGO falla:                                                           │
│     □ git checkout al commit anterior                                        │
│     □ Analizar causa del fallo                                               │
│     □ Ajustar implementación y repetir desde paso 2                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Resumen Final

### Optimizaciones Recomendadas para Implementación Inmediata

1. **P0-B (Buffers pre-alocados)**: Fácil, bajo riesgo, mejora de memoria
2. **P0-C (Compass cache)**: Alto impacto, bajo riesgo
3. **P1-B (Transformer cache)**: Trivial, 0 riesgo

### Optimizaciones que Requieren Testing Cuidadoso

1. **P0-A (map_coordinates unificado)**: Mayor impacto pero requiere validación visual
2. **P1-A (Compositing fusionado)**: Requiere validar orden de capas

### Optimizaciones Opcionales/Futuras

1. **P2-A (Supersampling adaptativo)**: Requiere cambios de interfaz
2. **P2-B (Float16)**: Beneficio menor, requiere validación de precisión

---

*Documento generado para implementación por agente AI.*
*Última actualización: 2026-01-29*
*Basado en análisis de: render_gpu.py, app.py, gpu_utils.py*
