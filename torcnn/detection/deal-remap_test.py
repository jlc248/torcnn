import pyart
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
sys.path.insert(0,'../')
sys.path.append('../vda')
import rad_utils
from datetime import datetime

# 1. Read the full NEXRAD L2 file
#radar = pyart.io.read_nexrad_archive('/myrorss2/data/thea.sandmael/data/radar/20151223/KNQA/raw/KNQA20151223_205305_V06.gz')
radar = pyart.io.read_nexrad_archive('KHGX20240105_115541_V06.gz')

#raddt = datetime(2015,12,12,20,53)
#t0=time.time()
#data = rad_utils.dealias_velocity_pyart(raddt, tilt=0.5, radar=radar)
#print(time.time()-t0)
#print(data.shape)
#plt.imshow(data)
#
#sys.exit()

# 2. Get the index for the 0.5-degree tilt (usually the first one, but it's good practice to verify)
sweep_number = 1
tilt_angle = radar.fixed_angle['data'][sweep_number]
print(f"Processing sweep at {tilt_angle:.2f} degrees.")

# 3. Create a new radar object containing only the selected sweep
radar_sweep = radar.extract_sweeps([sweep_number])

# --- Now, proceed with the quality control and dealiasing on the single-sweep radar object ---

# 4. Define the gate filter on the single sweep
gatefilter = pyart.filters.GateFilter(radar_sweep)

# A) Exclude gates with low reflectivity
if 'reflectivity' in radar_sweep.fields:
    gatefilter.exclude_below('reflectivity', 0.0)

# B) Exclude gates with low cross-correlation coefficient
#if 'cross_correlation_ratio' in radar_sweep.fields:
#    gatefilter.exclude_below('cross_correlation_ratio', 0.85)

# 5. Get the Nyquist velocity for this sweep
nyquist_vel = np.max(radar_sweep.instrument_parameters['nyquist_velocity']['data'][0])

# 6. Dealias the velocity field
t0=time.time()-t0
dealiased_velocity_field = pyart.correct.dealias_region_based(
    radar_sweep,
    vel_field='velocity',
    nyquist_vel=nyquist_vel,
    #gatefilter=gatefilter,
    centered=True,
    masked=False # can't regrid masked data
)
print('dealiased',time.time()-t0)

# 7. Add the new dealiased field back to the single-sweep radar object
radar_sweep.add_field('dealiased_velocity', dealiased_velocity_field, replace_existing=True)

data_array = radar_sweep.fields['dealiased_velocity']['data']

# Remapping

grid_shape = (1, 512, 512) # 20 vertical levels, 201 points in y and x
# Grid limits (minimum and maximum values in meters for z, y, x)
# Example: up to 10 km altitude, 100 km extent in x and y
z_limits = (0.0, 15000.0)
y_limits = (-160000.0, 160000.0)
x_limits = (-160000.0, 160000.0)
grid_limits = (z_limits, y_limits, x_limits)

t0=time.time()
grid = pyart.map.grid_from_radars(
    (radar_sweep,),
    grid_shape=grid_shape,
    grid_limits=grid_limits,
    fields=['reflectivity', 'dealiased_velocity'],
    gridding_algo='map_gates_to_grid', # 'map_gates_to_grid' is typically faster
    weighting_function='Cressman', # Choose a weighting function (e.g., 'Barnes', 'Cressman', 'Nearest')
)
print(time.time() - t0)

data = grid.fields['dealiased_velocity']['data']
print(f"Gridded data shape: {data.shape}")
#plt.imshow(data[0], vmin=-50, vmax=50)
#plt.savefig('tmp.png')
#print(data.max(),data.min())
#sys.exit()
# Plot a horizontal slice (e.g., the lowest level, index 0)
fig = plt.figure(figsize=(8, 6))
display = pyart.graph.GridMapDisplay(grid)
# Plot the lowest level (level=0)
display.plot_grid('dealiased_velocity', level=0, vmin=-50, vmax=50.0)
plt.savefig('tmp.png')

