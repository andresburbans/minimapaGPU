"""
Script de diagnóstico para verificar configuración de GPU y FFmpeg.
Ejecutar este script antes de iniciar el backend para ver el estado del sistema.
"""
import subprocess
import sys
from pathlib import Path

# Agregar el directorio backend al path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gpu_utils
    
    print("=" * 60)
    print("DIAGNÓSTICO DE GPU Y ACELERACIÓN")
    print("=" * 60)
    print()
    
    # Información de GPU
    gpu_info = gpu_utils.detect_cuda_gpu()
    print(gpu_utils.get_gpu_info_str())
    print()
    
    # Detalles técnicos
    print("Detalles técnicos:")
    print(f"  - CUDA disponible: {gpu_info['cuda_available']}")
    print(f"  - Número de GPUs: {gpu_info['gpu_count']}")
    print(f"  - GPU preferida: {gpu_info['preferred_gpu_id']}")
    print(f"  - NVENC disponible: {gpu_info['nvenc_available']}")
    print()
    
    # Verificar FFmpeg
    print("Verificando FFmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        first_line = result.stdout.split("\n")[0]
        print(f"✅ {first_line}")
        
        # Verificar codecs disponibles
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        
        codecs = []
        if "h264_nvenc" in result.stdout:
            codecs.append("h264_nvenc (GPU)")
        if "libx264" in result.stdout:
            codecs.append("libx264 (CPU)")
        if "hevc_nvenc" in result.stdout:
            codecs.append("hevc_nvenc (GPU)")
            
        print(f"   Codecs H.264 disponibles: {', '.join(codecs)}")
        
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"❌ Error verificando FFmpeg: {e}")
    
    print()
    
    # Verificar OpenCV
    print("Verificando OpenCV...")
    try:
        import cv2
        print(f"✅ OpenCV versión: {cv2.__version__}")
        
        # Verificar soporte CUDA en OpenCV
        if hasattr(cv2, "cuda"):
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                print(f"✅ OpenCV compilado con CUDA: {cuda_devices} dispositivo(s)")
                device_name = cv2.cuda.printShortCudaDeviceInfo(0)
            else:
                print("⚠️  OpenCV con CUDA pero sin dispositivos detectados")
        else:
            print("⚠️  OpenCV sin soporte CUDA (usando versión CPU)")
            
    except ImportError as e:
        print(f"❌ Error importando OpenCV: {e}")
    
    print()
    
    # Verificar otras dependencias
    print("Verificando otras dependencias...")
    dependencies = [
        "numpy",
        "pandas",
        "rasterio",
        "pillow",
        "shapely",
        "geopandas",
    ]
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "?")
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ❌ {dep}: No instalado")
    
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    
    if gpu_info["cuda_available"] and gpu_info["nvenc_available"]:
        print("🚀 SISTEMA ÓPTIMO: GPU y NVENC disponibles")
        print("   El sistema usará aceleración por hardware para:")
        print("   - Codificación de video (NVENC)")
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("   - Procesamiento de imagen (CUDA)")
    elif gpu_info["cuda_available"]:
        print("⚠️  GPU detectada pero NVENC no disponible")
        print("   Revisa los drivers de NVIDIA o la versión de FFmpeg")
    else:
        print("ℹ️  Modo CPU: Todo el procesamiento será en software")
        print("   Para mejor rendimiento, instala drivers NVIDIA y FFmpeg con NVENC")
    
    print()
    
except Exception as e:
    print(f"❌ Error en diagnóstico: {e}")
    import traceback
    traceback.print_exc()

print("Diagnóstico completado.")
print()
