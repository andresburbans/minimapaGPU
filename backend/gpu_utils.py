"""
Utilidades para detección y forzado de GPU NVIDIA usando CUDA.
Prioriza el uso de GPU dedicada (RTX 3050) sobre CPU para procesamiento.
"""
import os
import subprocess
import sys
import site
from pathlib import Path
from typing import Optional, Dict


def fix_cuda_dll_paths():
    """
    Agrega directorios de DLLs de NVIDIA al PATH y al search path de DLLs (Python 3.8+).
    Resuelve el error 'CuPy failed to load nvrtc64_120_0.dll'.
    """
    if sys.platform != "win32":
        return

    # Lista de DLLs críticas que buscamos (runtime y nvrtc)
    # A menudo estas son las que fallan con CuPy
    target_dlls = ["nvrtc64_120_0.dll", "cudart64_12.dll"]
    
    # Directorios donde buscar: site-packages, pero también la raíz del venv o sys.prefix
    search_roots = [sys.prefix]
    try:
        search_roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    if hasattr(site, 'getusersitepackages'):
        search_roots.append(site.getusersitepackages())

    # Eliminar duplicados y convertir a Path
    search_roots = list(set([Path(p) for p in search_roots]))
    
    added_paths = set()
    
    print(f"[GPU] Buscando librerías CUDA en: {[str(p) for p in search_roots]}...")

    # Estrategia 1: Buscar carpetas 'nvidia/*/bin' (estándar pip)
    for root in search_roots:
        # Posibles ubicaciones de la carpeta 'nvidia'
        possible_nvidia_dirs = [
            root / "nvidia",
            root / "Lib" / "site-packages" / "nvidia",
            root / "lib" / "site-packages" / "nvidia",
        ]
        
        for nvidia_base in possible_nvidia_dirs:
            if nvidia_base.exists() and nvidia_base.is_dir():
                try:
                    # Listamos carpetas de paquetes (cuda_runtime, nvrtc, etc)
                    for pkg_dir in nvidia_base.iterdir():
                        if pkg_dir.is_dir():
                            bin_dir = pkg_dir / "bin"
                            if bin_dir.exists() and bin_dir.is_dir():
                                path_str = str(bin_dir.absolute())
                                if path_str not in added_paths:
                                    try:
                                        os.add_dll_directory(path_str)
                                        if path_str not in os.environ["PATH"]:
                                            os.environ["PATH"] = path_str + os.pathsep + os.environ["PATH"]
                                        added_paths.add(path_str)
                                    except Exception:
                                        pass
                except Exception:
                    pass

    # Estrategia 2: Buscar en el root del venv (algunas instalaciones ponen DLLs en la raíz o en bin)
    for root in search_roots:
        for bin_name in ["bin", "Scripts"]:
            bin_dir = root / bin_name
            if bin_dir.exists() and bin_dir.is_dir():
                path_str = str(bin_dir.absolute())
                if path_str not in added_paths:
                    # Solo agregamos si contiene alguna de nuestras DLLs objetivo
                    has_target = any((bin_dir / dll).exists() for dll in target_dlls)
                    if has_target:
                        try:
                            os.add_dll_directory(path_str)
                            if path_str not in os.environ["PATH"]:
                                os.environ["PATH"] = path_str + os.pathsep + os.environ["PATH"]
                            added_paths.add(path_str)
                        except Exception:
                            pass
                
    if added_paths:
        print(f"[GPU] Se agregaron {len(added_paths)} directorios de DLLs de NVIDIA al search path.")
    else:
        # Si no encontramos nada con búsqueda rápida, avisar pero no bloquear con rglob
        print("[GPU] ADVERTENCIA: No se encontraron directorios de DLLs de NVIDIA mediante búsqueda rápida.")


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


_LAST_STATS = {"time": 0, "data": None}

def get_system_stats() -> Dict:
    """
    Obtiene estadísticas en tiempo real de CPU y GPU.
    Cacheado por 2 segundos para evitar saturar con llamadas a nvidia-smi.
    """
    import psutil
    import time
    
    global _LAST_STATS
    now = time.time()
    
    # Retornar cache si es reciente (< 2s)
    if _LAST_STATS["data"] and (now - _LAST_STATS["time"] < 2.0):
        return _LAST_STATS["data"]

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
    
    _LAST_STATS = {"time": now, "data": stats}
    return stats


# Inicializar GPU al importar el módulo
def _auto_init():
    """Auto-inicialización al importar."""
    try:
        fix_cuda_dll_paths()
        info = detect_cuda_gpu()
        if info["cuda_available"]:
            force_cuda_gpu()
            print(f"[GPU] Forzando uso de GPU: {info['gpu_names'][info['preferred_gpu_id']]}")
        else:
            print("[GPU] No se detectó GPU NVIDIA, usando CPU")
    except Exception as e:
        print(f"[GPU] Error en inicialización: {e}")
    finally:
        print("[GPU] Inicialización terminada.")

# Ejecutar auto-inicialización
_auto_init()
