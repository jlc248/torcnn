import pandas as pd
import numpy as np
import sys,os
from datetime import datetime
import glob
sys.path.insert(0, '../')
sys.path.append('../vda')
import pyproj
import rad_utils
import time

def latlon_to_yolo_label(
    radar_lat: float,
    radar_lon: float,
    obj_lat: float,
    obj_lon: float,
    image_shape: tuple[int, int] = (512,512),
    physical_lengths_in_km: tuple[float, float] = (320, 320),
    obj_box_size_in_km: tuple[int, int] = (10, 10),
    class_index: int = 0,
) -> str:

    """
    Converts radar-obj latitude/longitude to a normalized YOLO bounding box label.

    Args:
        radar_lat (float): Latitude of the radar site (center of the domain).
        radar_lon (float): Longitude of the radar site (center of the domain).
        obj_lat (float): Latitude of the object centroid.
        obj_lon (float): Longitude of the object centroid.
        image_shape (int,int): Shape of the Cartesian grid.
        physical_side_lengths_in_km (float, float): Total physical width/height of the domain in kilometers.
        obj_box_size_in_km (int, int): The fixed size (width and height) of the object bounding box in km.
        class_index (int): The integer class label for the object (default 0).

    Returns:
        str: A single line in the YOLO label format: "class_idx x_center_normed y_center_normed w h".
    """

    # Define the Coordinate Systems
    ## WGS84 (Standard Lat/Lon)
    wgs84_crs = pyproj.CRS.from_user_input('EPSG:4326')
    ## Azimuthal Equidistant
    local_proj_crs = pyproj.CRS.from_user_input({
        'proj':'aeqd', # Azimuthal Equidistant
        'lat_0':radar_lat, 
        'lon_0':radar_lon, 
        'units':'m'
    })
   
    # Create the Transformer object
    transformer = pyproj.Transformer.from_crs(wgs84_crs, local_proj_crs, always_xy=True)
 
    # Convert the object lat/lon to local physical units in meters
    X_obj_m, Y_obj_m = transformer.transform(obj_lon, obj_lat)

    # Map physical coordinates to pixel coordinates
    ## Convert parameters to meters
    L1, L2 = physical_lengths_in_km[0] * 1000.0, physical_lengths_in_km[1] * 1000.0 # Total domain side length in meters (e.g., 100000 m)
    R1 = L1 / image_shape[0] # Resolution in meters per pixel
    R2 = L2 / image_shape[1]

    # Calculate pixel indices (0 to image_size-1)
    ## The (L / 2) term shifts the origin from the radar center (0,0) to the array's corner.
    X_pixel = np.round((X_obj_m + L2 / 2) / R2).astype(int)
    Y_pixel = np.round((L1 / 2 - Y_obj_m) / R1).astype(int) # This is inverted so that (0,0) is NW corner of image.

    # Clamp to ensure coordinates are within the array bounds [0, image_size - 1]
    X_pixel = np.clip(X_pixel, 0, image_shape[1] - 1)
    Y_pixel = np.clip(Y_pixel, 0, image_shape[0] - 1)

    # Convert Pixel Centroid to YOLO Normalized Format
    ## Normalize centroid (x_center, y_center)
    x_center_norm = X_pixel / image_shape[1]
    y_center_norm = Y_pixel / image_shape[0]
  
    ## Normalize width and height
    obj_box_size_m_Y, obj_box_size_m_X = obj_box_size_in_km[0] * 1000.0, obj_box_size_in_km[1] * 1000.0
    
    w_norm = obj_box_size_m_X / L2
    h_norm = obj_box_size_m_Y / L1
   
    # Format the output as a string with high precision for coordinates
    yolo_label = f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
 
    return yolo_label



# Shape of the NEXRAD data when remapped to Cartesian coords
image_shape = (512, 512)

#datadir1 = '/myrorss2/work/thea.sandmael/radar' # 2019-2024
#datadir2 = '/myrorss2/data/thea.sandmael/data/radar' # 2011-2018

input_csv = '/raid/jcintineo/torcnn/torp_datasets/2024_Storm_Reports_Expanded_tilt0050_radar_r2500_nodup.csv'

outpatt = '/raid/jcintineo/torcnn/detection/truth_files/%Y/%Y%m%d/{radar}_%Y%m%d-%H%M%S.txt'

# Read radar xml
rad_dict = rad_utils.parse_radar_xml('../static/radarinfo.xml')

df = pd.read_csv(input_csv)

# For each TORP detect, add location to truth files
for row in df.itertuples(index=False):

    radar = row.radar

    if row.radar[0] != 'K':
        continue

    dt = datetime.strptime(row.radarTimestamp,'%Y%m%d-%H%M%S')

    lat = row.latitudeExtractCenter
    lon = row.longitudeExtractCenter

    if row.tornado > 0 and row.spout == 0: # exclude spouts for now 
        class_index = 0
    elif row.tornado == 0: # nontor
        class_index = 1
    else:
        continue
        
    label = latlon_to_yolo_label(rad_dict[radar]['lat'],
                               rad_dict[radar]['lon'],
                               lat,
                               lon,
                               image_shape=image_shape,
                               class_index=class_index
    ) 
    
    outfile = dt.strftime(outpatt.replace('{radar}', radar))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'a') as f:
        f.write(label + '\n')
