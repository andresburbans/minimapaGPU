import json
import urllib.request
import io
import os
import time
from pathlib import Path

def test_repro():
    # URL del servidor local
    url = "http://127.0.0.1:8000/preview"
    
    # Rutas absolutas correctas
    base = Path(r"d:\Dev\MinimapaGPU\backend\gpu_validation")
    ortho = base / "test_ortho_crop.tif"
    csv = base / "test_segments.csv"
    
    payload = {
        "config": {
            "ortho_path": str(ortho),
            "csv_path": str(csv),
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
        "time_sec": 0.5
    }

    def send_req(msg):
        print(f"\n[CLIENT] {msg}...")
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            t0 = time.time()
            with urllib.request.urlopen(req) as response:
                result = response.read()
                elapsed = time.time() - t0
                print(f"[CLIENT] Success! Time: {elapsed:.2f}s (Size: {len(result)} bytes)")
                return True
        except Exception as e:
            print(f"[CLIENT] ERROR: {e}")
            if hasattr(e, 'read'):
                print(f"[CLIENT] Detail: {e.read().decode()}")
            return False

    # 1. Preview 1 (Preloads GPU)
    send_req("Starting Preview 1 (GPU Preload)")
    
    # 2. Preview 2 (Uses Context)
    send_req("Starting Preview 2 (Context Reuse)")
    
    # 3. Simulate RENDER request (This starts a thread that calls preload_track_gpu)
    render_url = "http://127.0.0.1:8000/render"
    print("\n[CLIENT] Starting RENDER request...")
    try:
        req = urllib.request.Request(
            render_url, 
            data=json.dumps(payload['config']).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            job_id = res_data['job_id']
            print(f"[CLIENT] Render Job Started: {job_id}")
            
            # Pool status
            for _ in range(20):
                time.sleep(1)
                status_req = urllib.request.Request(f"http://127.0.0.1:8000/status?job_id={job_id}")
                with urllib.request.urlopen(status_req) as s_resp:
                    status = json.loads(s_resp.read().decode())
                    print(f"[CLIENT] Job Status: {status['status']} - {status['message']}")
                    if status['status'] in ['finished', 'failed', 'error']:
                        break
    except Exception as e:
        print(f"[CLIENT] Render Request Error: {e}")

if __name__ == "__main__":
    test_repro()
