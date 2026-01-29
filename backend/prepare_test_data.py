
import os
import shutil
import rasterio
from rasterio.windows import Window
import numpy as np

SOURCE_DIR = r"G:\VIDEO-RIVERA\Shapes\Geojsons"
DEST_DIR = r"d:\Dev\MinimapaGPU\backend\gpu_validation"
TIF_NAME = "LariveraCom_Trans2.tif"
GEOJSONS = ["Vias.geojson", "LinderoGeneral.geojson"]

def prepare_data():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        
    src_tif = os.path.join(SOURCE_DIR, TIF_NAME)
    dst_tif = os.path.join(DEST_DIR, "test_ortho_crop.tif")
    
    print(f"Opening {src_tif}...")
    try:
        with rasterio.open(src_tif) as src:
            w = src.width
            h = src.height
            print(f"Source Size: {w}x{h}")
            
            # Crop center 2048x2048
            cw, ch = 2048, 2048
            off_x = (w - cw) // 2
            off_y = (h - ch) // 2
            
            window = Window(off_x, off_y, cw, ch)
            
            print(f"Reading crop window: {window}")
            data = src.read(window=window)
            
            # Update transform
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update({
                'height': ch,
                'width': cw,
                'transform': transform,
                'compress': 'lzw',
                'tiled': True
            })
            
            print(f"Saving to {dst_tif}...")
            with rasterio.open(dst_tif, 'w', **profile) as dst:
                dst.write(data)
                
    except Exception as e:
        print(f"Error processing TIF: {e}")

    # Copy GeoJSONs
    for g in GEOJSONS:
        src_g = os.path.join(SOURCE_DIR, g)
        dst_g = os.path.join(DEST_DIR, g)
        print(f"Copying {g}...")
        try:
            shutil.copy2(src_g, dst_g)
        except Exception as e:
            print(f"Error copying {g}: {e}")

if __name__ == "__main__":
    prepare_data()
