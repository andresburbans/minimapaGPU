# Instalación de FFmpeg con Soporte NVENC

## 🎯 ¿Por qué necesitas esto?

NVENC es la tecnología de codificación de video por hardware de NVIDIA. Con NVENC:
- ⚡ **30-50x más rápido** que codificación en CPU
- 💻 **Libera la CPU** para otras tareas
- 🎬 **Misma calidad** de video
- 🔋 **Menor consumo energético**

## 📥 Descargar FFmpeg con NVENC

### Opción 1: Build Oficial (Recomendado)

1. Ve a: https://github.com/BtbN/FFmpeg-Builds/releases

2. Descarga el archivo más reciente que contenga:
   - `ffmpeg-n*-**win64-gpl-shared**-*.zip`
   
   Ejemplo: `ffmpeg-n6.1-latest-win64-gpl-shared-6.1.zip`

3. Descomprime el archivo

4. Dentro encontrarás una carpeta `bin/` con `ffmpeg.exe`

### Opción 2: Build de gyan.dev

1. Ve a: https://www.gyan.dev/ffmpeg/builds/

2. Descarga: **ffmpeg-release-full.7z**

3. Descomprime el archivo

## ⚙️ Instalación

### Opción A: Reemplazar FFmpeg Global

1. Localiza tu FFmpeg actual:
   ```powershell
   where.exe ffmpeg
   ```

2. Respalda el FFmpeg actual (por si acaso):
   ```powershell
   move "C:\ruta\a\ffmpeg.exe" "C:\ruta\a\ffmpeg.exe.backup"
   ```

3. Copia el nuevo `ffmpeg.exe` a la misma ubicación

### Opción B: Agregar a PATH (Recomendado)

1. Copia la carpeta descomprimada a una ubicación permanente:
   ```
   C:\FFmpeg\
   ```

2. Agrega `C:\FFmpeg\bin` a la variable PATH:
   - Presiona `Win + R`
   - Escribe: `sysdm.cpl` y Enter
   - Ve a: **Opciones avanzadas** → **Variables de entorno**
   - En **Variables del sistema**, selecciona `Path` → **Editar**
   - Agrega nueva entrada: `C:\FFmpeg\bin`
   - Click **OK** en todas las ventanas

3. **IMPORTANTE:** Cierra y reabre PowerShell/CMD

## ✅ Verificar Instalación

```powershell
# Verificar que FFmpeg se encuentra
ffmpeg -version

# Verificar que NVENC está disponible
ffmpeg -hide_banner -encoders | findstr nvenc
```

Deberías ver:
```
V....D h264_nvenc           NVIDIA NVENC H.264 encoder
V....D hevc_nvenc           NVIDIA NVENC hevc encoder
```

## 🔍 Verificar en MinimapaGPT

Desde el directorio del proyecto:

```powershell
cd D:\Dev\MinimapaGPT\backend
.venv\Scripts\activate
python diagnostico.py
```

Si todo está correcto, verás:
```
✅ GPU NVIDIA detectada: 1 dispositivo(s)
🎯 GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU
✅ NVENC (codificación hardware) disponible
```

## 🚀 Resultados

**Antes (sin NVENC):**
- Codificación: libx264 (CPU)
- Tiempo: ~5-10 minutos para video de 2 min
- Uso CPU: 80-100%

**Después (con NVENC):**
- Codificación: h264_nvenc (GPU)
- Tiempo: ~10-30 segundos para video de 2 min
- Uso CPU: 10-20%
- Uso GPU: 40-60%

## ❓ Troubleshooting

### "NVENC no disponible" después de instalar

1. **Reinicia PowerShell/CMD** (importante)
2. Verifica que `ffmpeg -version` muestre la nueva versión
3. Verifica drivers NVIDIA actualizados:
   ```powershell
   nvidia-smi
   ```

### "Cannot load nvcuda.dll"

- Actualiza drivers NVIDIA desde: https://www.nvidia.com/Download/index.aspx
- GPU: RTX 3050 Laptop
- OS: Windows 11

### FFmpeg no encontrado después de cambiar PATH

- Cierra **todas** las ventanas de PowerShell/CMD
- Abre una nueva ventana
- Verifica con: `where.exe ffmpeg`

## 📝 Notas

- NVENC está disponible en GPUs NVIDIA desde GTX 600 series en adelante
- RTX 3050 tiene NVENC de 8va generación (excelente calidad)
- El sistema automáticamente usa NVENC si está disponible
- Si NVENC no está, usa libx264 automáticamente (sin errores)
