
import sys
import os
sys.path.append(os.path.abspath('backend'))
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from render import render_frame

# Create a dummy GeoTIFF
data = np.zeros((3, 100, 100), dtype=np.uint8)
transform = from_origin(4400000, 2000000, 1, 1)
with rasterio.open('test.tif', 'w', driver='GTiff', height=100, width=100, count=3, dtype='uint8', crs='EPSG:9377', transform=transform) as ds:
    ds.write(data)

with rasterio.open('test.tif') as dataset:
    vectors = []
    center_e, center_n = 4400050, 2000050
    heading = 0
    width, height = 1920, 1080
    map_half_width_m = 150
    arrow_size_px = 400
    cone_angle_deg = 60
    cone_length_px = 220
    cone_opacity = 0.18
    icon_circle_opacity = 0.35
    icon_circle_size_px = 120
    
    try:
        frame = render_frame(
            dataset, vectors, center_e, center_n, heading,
            width, height, map_half_width_m, arrow_size_px,
            cone_angle_deg, cone_length_px, cone_opacity,
            icon_circle_opacity, icon_circle_size_px
        )
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()
