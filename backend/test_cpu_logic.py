
import os
import sys
import numpy as np
from PIL import Image
import math
from rasterio.transform import Affine

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import render
    print("✅ Port 'render' exitoso")
except Exception as e:
    print(f"❌ Error al importar 'render': {e}")
    sys.exit(1)

def test_to_rgba_cpu():
    print("Testing _to_rgba (CPU)...")
    # Simulate a 3-channel (RGB) dataset read: (3, 10, 10)
    data = np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8)
    rgb, alpha = render._to_rgba(data)
    if rgb.shape == (10, 10, 3) and alpha.shape == (10, 10):
        print("   ✅ _to_rgba (RGB) OK")
    else:
        print(f"   ❌ _to_rgba (RGB) Falló: {rgb.shape}, {alpha.shape}")

def test_normalize_rgba_cpu():
    print("Testing _normalize_rgba (CPU)...")
    rgb = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    alpha = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    norm = render._normalize_rgba(rgb, alpha)
    if norm.shape == (10, 10, 4) and norm.dtype == np.uint8:
        print("   ✅ _normalize_rgba OK")
    else:
        print(f"   ❌ _normalize_rgba Falló: {norm.shape}")

def test_coordinate_math_cpu():
    print("Testing _latlon_to_pixel (CPU)...")
    # Test a known point (0,0) at zoom 0
    x, y = render._latlon_to_pixel(0, 0, 0)
    # At zoom 0, the world is 256x256. (0,0) is center (128, 128)
    if math.isclose(x, 128) and math.isclose(y, 128):
        print("   ✅ _latlon_to_pixel OK")
    else:
        print(f"   ❌ _latlon_to_pixel Falló: ({x}, {y})")

if __name__ == "__main__":
    print("Iniciando Pruebas de Lógica CPU...")
    test_to_rgba_cpu()
    test_normalize_rgba_cpu()
    test_coordinate_math_cpu()
    print("\n✅ Todas las pruebas de lógica CPU completadas.")
