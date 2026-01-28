#!/usr/bin/env python
"""Test script para verificar que todo está funcionando correctamente"""

import sys
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("PRUEBA FINAL DEL SISTEMA")
print("=" * 60)
print()

try:
    # Test 1: Importar gpu_utils
    print("1. Importando gpu_utils...")
    import gpu_utils
    print("   ✅ gpu_utils importado")
    
    # Test 2: Importar track
    print("2. Importando track...")
    import track
    print("   ✅ track importado")
    
    # Test 3: Importar app
    print("3. Importando app...")
    import app
    print("   ✅ app importado")
    
    print()
    print("-" * 60)
    print("INFORMACIÓN DE GPU")
    print("-" * 60)
    
    # Test 4: Verificar GPU
    gpu_info = gpu_utils.detect_cuda_gpu()
    
    print(f"CUDA disponible: {gpu_info['cuda_available']}")
    print(f"Número de GPUs: {gpu_info['gpu_count']}")
    print(f"GPU preferida: {gpu_info['preferred_gpu_id']}")
    print(f"NVENC disponible: {gpu_info['nvenc_available']}")
    
    if gpu_info['gpu_names']:
        print(f"\nGPUs detectadas:")
        for idx, name in enumerate(gpu_info['gpu_names']):
            marker = "🎯" if idx == gpu_info['preferred_gpu_id'] else "  "
            print(f"  {marker} GPU {idx}: {name}")
    
    print()
    print("-" * 60)
    print("ESTADO FINAL")
    print("-" * 60)
    
    if gpu_info['cuda_available']:
        print("✅ GPU detectada y configurada")
        if gpu_info['nvenc_available']:
            print("✅ NVENC disponible - Codificación GPU acelerada")
            print("🚀 SISTEMA ÓPTIMO")
        else:
            print("⚠️  NVENC no disponible - Codificación en CPU")
            print("💡 Instala FFmpeg con NVENC para mejor rendimiento")
            print("   Ver: INSTALACION_NVENC.md")
    else:
        print("ℹ️  GPU no detectada - Modo CPU")
    
    print("\n✅ TODOS LOS TESTS PASARON")
    print()
    print("-" * 60)
    print("PARA INICIAR EL SERVIDOR:")
    print("-" * 60)
    print("uvicorn app:app --reload --port 8000")
    print()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
