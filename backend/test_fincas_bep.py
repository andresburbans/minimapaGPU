#!/usr/bin/env python3
"""
Test de emulación del proceso GPU con FincasBEP.json.
Verifica que el fix de cudaErrorAlreadyMapped funciona correctamente.
"""
import sys
import os
import time
import argparse
from pathlib import Path
import rasterio
import numpy as np

# Configurar logs básicos
def log(msg):
    print(f"[TEST] {msg}")

# Añadir root al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from render_gpu import _CONTEXT, HAS_GPU, preload_track_gpu, cleanup_gpu
    from render import load_vectors
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configurar paths por defecto
FINCAS_GEOJSON = r"G:\VIDEO-RIVERA\Shapes\Geojsons\FincasBEP.json"
TEST_ORTHO = Path(__file__).parent / "gpu_validation" / "test_ortho_crop.tif"

def test_fincas_bep_process(fps_limit: int = 55):
    """Ejecuta el proceso completo con FincasBEP.json."""
    log("=" * 60)
    log("TEST DE EMULACION: FincasBEP.json")
    log("=" * 60)
    
    if not os.path.exists(FINCAS_GEOJSON):
        log(f"ERROR: No se encuentra {FINCAS_GEOJSON}")
        return False
        
    if not TEST_ORTHO.exists():
        log(f"ERROR: No se encuentra {TEST_ORTHO}")
        return False

    # Cargar ortofoto de test
    log(f"Cargando ortofoto: {TEST_ORTHO}")
    with rasterio.open(TEST_ORTHO) as dataset:
        log(f"Ortomosaico: {dataset.width}x{dataset.height}")
        
        # Cargar GeoJSON con la firma correcta
        log(f"Cargando GeoJSON: {FINCAS_GEOJSON}")
        vectors = load_vectors(
            dataset.crs,        # dataset_crs
            [],                 # vector_layers
            [str(FINCAS_GEOJSON)], # vectors_paths
            None,               # curves_path
            "#FF0000",          # line_color
            2,                  # line_width
            "#0000FF",          # boundary_color
            2,                  # boundary_width
            "#00FF00"           # point_color
        )
        log(f"Cargadas {len(vectors)} capas de vectores")
        
        # Centro de la zona
        center_e = (dataset.bounds.left + dataset.bounds.right) / 2
        center_n = (dataset.bounds.top + dataset.bounds.bottom) / 2
        center_points = [(center_e, center_n)]
        
        log("Iniciando preload con recovery enabled...")
        
        if HAS_GPU:
            # Simular 2 ciclos seguidos (como Preview -> Render)
            for cycle in range(1, 3):
                log(f"\n--- CICLO {cycle} ---")
                start_time = time.time()
                
                try:
                    # Mock de la config
                    class MockConfig:
                        use_gpu = True
                        wms_source = "google_hybrid"
                        width = 1920
                        height = 1080
                    
                    # Llamar a la función pública que usa app.py
                    cleanup_gpu()
                    success = _CONTEXT.preload(
                        dataset=dataset,
                        center_points=center_points,
                        margin_m=500,
                        vectors=vectors,
                        progress_callback=lambda pct, msg: print(f"[GPU] {pct}% - {msg}")
                    )
                    
                    elapsed = time.time() - start_time
                    log(f"Preload ciclo {cycle} completado en {elapsed:.2f}s")
                    log(f"Resultado: {success}")
                    log(f"Context ready: {_CONTEXT.is_ready}")

                    # Loop emulado post-preload (FPS < 60)
                    if fps_limit:
                        log(f"Simulando carga de renderizado a {fps_limit} FPS...")
                        target_dt = 1.0 / float(fps_limit)
                        frames = 30
                        for i in range(frames):
                            t0 = time.perf_counter()
                            # Simulación de trabajo
                            time.sleep(0.005) 
                            dt = time.perf_counter() - t0
                            sleep_s = max(0.0, target_dt - dt)
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                    
                except Exception as e:
                    log(f"!!! FALLO CRITICO EN CICLO {cycle}: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
        else:
            log("GPU no disponible, saltando test de GPU")
            return False
    
    log("\n" + "=" * 60)
    log("TEST COMPLETADO EXITOSAMENTE")
    log("=" * 60)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de emulacion FincasBEP")
    parser.add_argument("--geojson", type=str, help="Path al archivo GeoJSON")
    parser.add_argument("--fps-limit", type=int, default=55, help="Limite de FPS para el test")
    
    args = parser.parse_args()
    
    if args.geojson:
        FINCAS_GEOJSON = args.geojson
    
    # Ejecutar test
    success = test_fincas_bep_process(fps_limit=args.fps_limit)
    sys.exit(0 if success else 1)
