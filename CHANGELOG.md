# Changelog - Optimizaciones GPU y Calidad de Video

## 2026-01-22 - Mejoras Críticas de Rendimiento y Calidad

### 🚀 Aceleración GPU Implementada

#### **Nuevo Módulo: `gpu_utils.py`**
- Detección automática de GPU NVIDIA
- Prioriza RTX 3050 si está disponible
- Configura variables de entorno CUDA automáticamente
- Verifica disponibilidad de NVENC (codificación hardware)

#### **Características:**
- Auto-inicialización al importar
- Forzado de GPU mediante `CUDA_VISIBLE_DEVICES`
- Soporte para OpenCV con CUDA
- Endpoint `/gpu-info` para verificar estado en tiempo real

### 🎬 Arreglo del Lag en Modo Path Aéreo

#### **Problema Original:**
El video exportado se veía "horrible, con lag o smoothing" debido a:
1. Codec `mp4v` (obsoleto y de baja calidad)
2. Sin control de frame rate constante
3. Mala configuración de FFmpeg

#### **Solución Implementada:**

**1. Cambio de Arquitectura**
- Antes: Escribir frames directamente con `cv2.VideoWriter` (mp4v)
- Ahora: Exportar frames PNG → FFmpeg con NVENC/libx264

**2. Flags de FFmpeg Optimizados**
```bash
-vsync cfr          # Constant Frame Rate (elimina lag)
-movflags +faststart # Optimización para streaming
-pix_fmt yuv420p    # Compatibilidad universal
```

**3. Codecs de Alta Calidad**
- **GPU (NVENC):** CQ 20, preset p5, VBR
- **CPU (libx264):** CRF 20, preset medium

### 📊 Mejoras en Ambos Modos

#### **Modo Minimapa**
- Usa NVENC si está disponible
- Fallback automático a libx264
- Mismos flags de calidad optimizados

#### **Modo Path Aéreo** 
- Renderizado frame-by-frame optimizado
- Codificación con GPU acelerada
- Limpieza automática de archivos temporales
- Mejor logging y manejo de errores

### 🔧 Mejoras Técnicas

#### **`app.py`**
- Import de `gpu_utils` para auto-configuración
- Endpoint `/gpu-info` agregado
- Endpoint `/health` mejorado con estado de GPU
- Función `_encode_video()` optimizada

#### **`track.py`**
- `render_overlay()` reescrito completamente
- Nueva función `_encode_overlay_video()` con soporte GPU
- Frames guardados como PNG (sin pérdida)
- Codificación final con FFmpeg optimizado

#### **`gpu_utils.py`** (nuevo)
- `detect_cuda_gpu()`: Detección completa de GPU
- `force_cuda_gpu()`: Forzar uso de GPU específica
- `get_gpu_info_str()`: Información legible
- Auto-inicialización al importar

### 📝 Documentación

#### **`diagnostico.py`** (nuevo)
Script de diagnóstico completo que verifica:
- GPUs detectadas y configuradas
- FFmpeg y codecs disponibles
- OpenCV con/sin CUDA
- Todas las dependencias

#### **`README.md`** (actualizado)
- Sección de aceleración GPU
- Instrucciones de verificación
- Troubleshooting completo
- Explicación de optimizaciones

### ⚙️ Configuración Automática

El sistema ahora:
1. Detecta GPUs al iniciar
2. Prioriza RTX 3050 automáticamente
3. Configura CUDA_VISIBLE_DEVICES
4. Selecciona mejor codec disponible
5. Muestra estado en consola

### 🎯 Resultados

**Antes:**
- ❌ Todo en CPU
- ❌ Video con lag/smoothing
- ❌ Codec mp4v de baja calidad
- ❌ Sin aprovechamiento de GPU RTX 3050

**Ahora:**
- ✅ GPU RTX 3050 forzada automáticamente
- ✅ NVENC (hardware encoding) cuando disponible
- ✅ Video fluido sin lag (vsync cfr)
- ✅ Alta calidad (CQ/CRF 20)
- ✅ Path overlay perfectamente sincronizado

### 🔍 Verificación

Ejecutar diagnóstico:
```powershell
cd D:\Dev\MinimapaGPT\backend
python diagnostico.py
```

Salida esperada:
```
[GPU] Forzando uso de GPU: NVIDIA GeForce RTX 3050 Laptop GPU
✅ GPU NVIDIA detectada: 1 dispositivo(s)
🎯 GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU
✅ NVENC (codificación hardware) disponible
```

### ⚠️ Notas Importantes

1. **FFmpeg con NVENC:** Si no tienes FFmpeg con soporte NVENC, el sistema usará automáticamente libx264 (CPU) con la misma calidad.

2. **Drivers NVIDIA:** Asegúrate de tener drivers actualizados para máximo rendimiento.

3. **OpenCV con CUDA:** La versión actual de OpenCV puede no tener soporte CUDA compilado. Esto no afecta la codificación de video (que usa FFmpeg + NVENC).

### 🚦 Estado de Implementación

- [x] Detección automática de GPU
- [x] Forzado de RTX 3050
- [x] NVENC para modo minimapa
- [x] NVENC para modo path aéreo
- [x] Flags FFmpeg optimizados
- [x] Arreglo de lag/smoothing
- [x] Limpieza de archivos temporales
- [x] Documentación completa
- [x] Script de diagnóstico
- [x] Endpoint /gpu-info
- [x] Logging mejorado

### 📈 Próximas Mejoras Posibles

- [ ] OpenCV compilado con CUDA para procesamiento de frames
- [ ] Soporte para múltiples GPUs
- [ ] Benchmark de rendimiento CPU vs GPU
- [ ] Cache de frames para preview más rápido
- [ ] Compresión temporal de archivos intermedios
