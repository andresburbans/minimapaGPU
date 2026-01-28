"""
Utilidades para detección y forzado de GPU NVIDIA usando CUDA.
Prioriza el uso de GPU dedicada (RTX 3050) sobre CPU para procesamiento.
"""
import os
import subprocess
from typing import Optional, Dict


_GPU_INFO: Optional[Dict] = None


def detect_cuda_gpu() -> Dict:
    """
    Detecta GPUs NVIDIA disponibles y configura CUDA.
    Intenta forzar el uso de GPU dedicada RTX 3050.
    """
    global _GPU_INFO
    
    if _GPU_INFO is not None:
        return _GPU_INFO
    
    info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "preferred_gpu_id": None,
        "nvenc_available": False,
    }
    
    # Verificar si nvidia-smi está disponible
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        
        lines = result.stdout.strip().split("\n")
        info["gpu_count"] = len(lines)
        
        for idx, line in enumerate(lines):
            gpu_name = line.split(",")[0].strip()
            info["gpu_names"].append(gpu_name)
            
            # Priorizar RTX 3050 si está presente
            if "rtx" in gpu_name.lower() and "305" in gpu_name.lower():
                info["preferred_gpu_id"] = idx
                info["cuda_available"] = True
        
        # Si no encontró RTX 3050, usar la primera GPU
        if info["preferred_gpu_id"] is None and info["gpu_count"] > 0:
            info["preferred_gpu_id"] = 0
            info["cuda_available"] = True
            
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Verificar si FFmpeg tiene soporte NVENC
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        info["nvenc_available"] = "h264_nvenc" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    _GPU_INFO = info
    return info


def force_cuda_gpu(gpu_id: Optional[int] = None) -> bool:
    """
    Fuerza el uso de GPU CUDA específica mediante variables de entorno.
    Si gpu_id es None, usa la GPU preferida detectada.
    Retorna True si se configuró correctamente.
    """
    gpu_info = detect_cuda_gpu()
    
    if not gpu_info["cuda_available"]:
        return False
    
    # Determinar qué GPU usar
    target_gpu = gpu_id if gpu_id is not None else gpu_info["preferred_gpu_id"]
    
    if target_gpu is None:
        return False
    
    # Configurar variables de entorno para CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Forzar uso de GPU para OpenCV si está compilado con CUDA
    try:
        import cv2
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)  # Usar el primer device visible (ya filtrado por CUDA_VISIBLE_DEVICES)
    except (ImportError, AttributeError):
        pass
    
    return True


def get_system_stats() -> Dict:
    """
    Obtiene estadísticas en tiempo real de CPU y GPU.
    """
    import psutil
    
    stats = {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "ram_usage": psutil.virtual_memory().percent,
        "gpu_available": False,
        "gpu_usage": 0,
        "gpu_temp": 0,
        "gpu_name": "N/A",
        "nvenc_available": False
    }
    
    gpu_info = detect_cuda_gpu()
    stats["nvenc_available"] = gpu_info["nvenc_available"]
    
    if gpu_info["cuda_available"]:
        stats["gpu_available"] = True
        preferred_id = gpu_info["preferred_gpu_id"]
        if preferred_id is not None:
            stats["gpu_name"] = gpu_info["gpu_names"][preferred_id]
            try:
                # Consultar uso de GPU específico
                result = subprocess.run(
                    [
                        "nvidia-smi", 
                        f"--id={preferred_id}", 
                        "--query-gpu=utilization.gpu,temperature.gpu", 
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2
                )
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    stats["gpu_usage"] = float(parts[0].strip())
                    stats["gpu_temp"] = float(parts[1].strip())
            except Exception:
                pass
                
    return stats


# Inicializar GPU al importar el módulo
def _auto_init():
    """Auto-inicialización al importar."""
    info = detect_cuda_gpu()
    if info["cuda_available"]:
        force_cuda_gpu()
        print(f"[GPU] Forzando uso de GPU: {info['gpu_names'][info['preferred_gpu_id']]}")
    else:
        print("[GPU] No se detectó GPU NVIDIA, usando CPU")


# Ejecutar auto-inicialización
_auto_init()
