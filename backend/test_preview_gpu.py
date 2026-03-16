import json
import urllib.request
import io
import os
from pathlib import Path

def test_preview_gpu():
    url = "http://127.0.0.1:8000/preview"
    
    # Use real test data that exists in the current repo
    payload = {
        "config": {
            "ortho_path": str(Path("gpu_validation/test_ortho_crop.tif").absolute()),
            "csv_path": str(Path("gpu_validation/test_segments.csv").absolute()),
            "vector_layers": [],
            "vectors_paths": [],
            "wms_source": "google_hybrid",
            "map_half_width_m": 50.0,
            "width": 1280,
            "height": 720,
            "duration_sec": 10,
            "use_gpu": True,
            "show_compass": True,
            "compass_size_px": 40,
            "arrow_size_px": 100,
            "cone_length_px": 200,
            "icon_circle_opacity": 0.4
        },
        "time_sec": 2.0
    }

    print(f"Enviando petición de Preview GPU para {Path(payload['config']['ortho_path']).name}...")
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        import time
        t0 = time.time()
        with urllib.request.urlopen(req) as response:
            result = response.read()
            elapsed = time.time() - t0
            print(f"Preview GPU Exitosa! Tiempo: {elapsed:.2f}s (Tamaño: {len(result)} bytes)")
            
            with open("preview_gpu_test.png", "wb") as f:
                f.write(result)
            print("Imagen guardada en preview_gpu_test.png")
            
        # Second request to test cache
        print("\nRe-enviando misma petición (Caché)...")
        t1 = time.time()
        with urllib.request.urlopen(req) as response:
            result = response.read()
            elapsed_cache = time.time() - t1
            print(f"Preview GPU (Cache) Exitosa! Tiempo: {elapsed_cache:.2f}s")
            
    except Exception as e:
        print(f"Error en Preview GPU: {e}")
        if hasattr(e, 'read'):
            print(f"Detalle: {e.read().decode()}")

if __name__ == "__main__":
    test_preview_gpu()
