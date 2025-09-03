from typing import Dict, List, Any
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from NWSColorMaps import NWSColorMaps
from geopy.distance import geodesic
from geopy.point import Point
import math
import os,sys
import glob
NWScmap = NWSColorMaps()

def plot_cartesian(file_path,
                   varname,
                   rangemax=300,
                   Xlat=None,
                   Xlon=None,
):
    """
    Plots WDSS2-decoded netcdfs im cartesian coords.
    Args:
    - file_path (str): full path to the netcdf
    - varname (str): variable name
    - rangemax (int): the maximum range to plot, in km from radar
    - Xlat (float or None): An optional latitude to plot
    - Xlon (float or None): An optional longitude to plot
    """

    # Open the netCDF file using xarray
    ds = xr.open_dataset(file_path)

    # Extract the data
    raddata = ds[varname].values
    azimuth = ds['Azimuth'].values
    gate_width = ds['GateWidth'].values.mean() # Assuming GateWidth is relatively constant
    range_to_first_gate = ds.attrs['RangeToFirstGate']
    radar_name = ds.attrs['radarName-value']

    # Create an array of gate distances (r)
    gates = np.arange(ds.dims['Gate'])
    r_meters = range_to_first_gate + (gates * gate_width)

    # Convert azimuth from degrees to radians for Matplotlib's polar plot
    # Azimuth angles typically go from 0 to 360.
    theta = np.deg2rad(azimuth)

    # Handle potential missing data values if needed
    # The ncdump shows MissingData = -99900.f and RangeFolded = -99901.f
    missing_data_value = ds.attrs.get('MissingData', -99900.0)
    range_folded_value = ds.attrs.get('RangeFolded', -99901.0)

    # Replace missing values with NaN for plotting
    rad_masked = np.where(raddata == missing_data_value, np.nan, raddata)
    rad_masked = np.where(rad_masked == range_folded_value, np.nan, rad_masked)

    # Find the index where the range exceeds the limit
    max_range_index = np.where(r_meters > rangemax * 1000)[0] # Convert km to meters for comparison

    if len(max_range_index) > 0:
        # Use the index of the first gate that exceeds the limit
        end_gate_index = max_range_index[0]
        r_meters_limited = r_meters[:end_gate_index]
        rad_masked_limited = rad_masked[:, :end_gate_index]
    else:
        # If all gates are within the limit, use the full data
        r_meters_limited = r_meters
        rad_masked_limited = rad_masked

    # Convert the limited range to kilometers
    r_km_limited = r_meters_limited / 1000.0

    R, THETA = np.meshgrid(r_km_limited, theta)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set some plotting config
    if varname == 'Reflectivity':
        cmap = NWScmap; vmin=-10; vmax=75
    elif varname == 'AzShear':
        cmap = 'bwr'; vmin=-0.006; vmax=0.006
    elif varname == 'DivShear':
        cmap = 'PiYG'; vmin=-0.006; vmax=0.006
    elif varname == 'Velocity' or varname == 'AliasedVelocity':
        cmap = 'PiYG'; vmin=-50; vmax=50 

    c = ax.pcolormesh(THETA, R, rad_masked_limited, cmap=cmap, vmin=vmin, vmax=vmax)
   
    # Add a colorbar
    cbar = fig.colorbar(c, ax=ax, orientation='vertical', pad=0.1, shrink=0.75)
    cbar.set_label(f'{varname} ({ds[varname].units})', fontsize=14) 
   
    plt.show()
#-----------------------------------------------------------------------------------------------
def get_img_info(c):
    return {
        'Velocity':{'cmap':'PiYG', 'vmin':-50, 'vmax':50, 'units':'m/s'},
        'AliasedVelocity':{'cmap':'PiYG', 'vmin':-50, 'vmax':50, 'units':'m/s'},
        'PhiDP':{'cmap':'gnuplot', 'vmin':0, 'vmax':360, 'units':'deg'},
        'RhoHV':{'cmap':'rainbow', 'vmin':0.45, 'vmax':1, 'units':'correlation'},
        'SpectrumWidth':{'cmap':'cubehelix', 'vmin':0, 'vmax':10, 'units':'m/s'},
        'AzShear':{'cmap':'bwr', 'vmin':-0.02, 'vmax':0.02, 'units':'1/s'},
        'DivShear':{'cmap':'PiYG', 'vmin':-0.02, 'vmax':0.02, 'units':'1/s'},
        'Reflectivity':{'cmap':NWScmap, 'vmin':-10, 'vmax':75, 'units':'dBZ'},
        'Zdr':{'cmap':'Spectral_r', 'vmin':-5, 'vmax':5, 'units':'dB'},
    }.get(c,'')

#-----------------------------------------------------------------------------------------------
def plot_radar(
        data: Dict[str,Any],
        channels: List[str]=['Reflectivity','Velocity'],
        fig:plt.Figure=None,
        include_cbar:bool=False,
        include_title:bool=True,
        n_rows:int=None,
        n_cols:int=None,
        full_ppi: bool=False,
):
    """
    Plot radar PPIs. Adapted from MIT LL TorNet code.
    """

    if fig is None:
        fig = plt.figure()
    
    if not full_ppi: 
        if data['az_lower'] > data['az_upper']:
            az_lower_to_use=data['az_lower']-360
        else:
            az_lower_to_use=data['az_lower']
    
        az_min  = np.float32(az_lower_to_use) * np.pi/180
        az_max = np.float32(data['az_upper']) * np.pi/180
        rmin = np.float32(data['rng_lower'])
        rmax = np.float32(data['rng_upper'])
    
        n_az_patch, n_rng_patch = data['Velocity']['data'].shape
    
        T = np.linspace(az_min,az_max,n_az_patch)
        R = np.linspace(rmin,rmax,n_rng_patch)
        R,T = np.meshgrid(R,T)


    for k, c in enumerate(channels):
        if n_rows is None:
            ax = fig.add_subplot(1, len(channels), k+1, polar=True)
        else:
            ax = fig.add_subplot(n_rows, n_cols, k+1, polar=True)

        ax.set_theta_zero_location('N') # radar convention
        ax.set_theta_direction(-1)

        ax.grid(False)
        
        info = get_img_info(c)
        cmap = info['cmap']
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        if c in ['Reflectivity', 'Zdr', 'RhoHV', 'PhiDP']:
            cmap.set_bad(color='white') 
        elif c in ['Velocity', 'AliasedVelocity', 'SpectrumWidth', 'AzShear', 'DivShear']:
            cmap.set_bad(color='purple')

        if full_ppi:
            R, T = np.meshgrid(data[c]['ranges'], np.deg2rad(data[c]['azimuths']))
            im = ax.pcolormesh(T, R, data[c]['data'], shading='nearest', cmap=cmap, vmin=info['vmin'], vmax=info['vmax'])
        else:
            im = ax.pcolormesh(T, R-rmin, data[c]['data'], shading='nearest', cmap=cmap, vmin=info['vmin'], vmax=info['vmax'])
            ax.set_rorigin(-rmin)
            ax.set_thetalim([az_min,az_max])


        ax.set_xticklabels([]) # turns off ticks
        ax.set_yticklabels([])

        fs = 6
        if full_ppi:
            fig.canvas.draw()
            rt = [75, 150, 225, 300]
            rt_labs = [str(_) for _ in rt]
            rt_labs[-1] += ' km'
            ax.set_rgrids(rt, labels=rt_labs, fontsize=fs)
        else:
            rt = np.linspace(0, rmax-rmin, 4)
            # This `fig.canvas.draw()` is needed to adjust fontsize due to some issue with matplotlib
            #https://github.com/matplotlib/matplotlib/issues/17463
            fig.canvas.draw() 
            rt_labs= [str(int(_ + rmin)) for _ in rt]
            rt_labs[0] += ' km'
            ax.set_rgrids(rt ,labels=rt_labs, fontsize=fs)

        if full_ppi:
            tt = [90, 180, 270, 360]
            ax.set_thetagrids(tt, labels=[str(_) + '$^\circ$' for _ in tt], fontsize=fs)
        else: 
            tt = np.linspace(az_lower_to_use, data['az_upper'], 4)
            tt_labs = []
            for theta_lab in tt:
                if theta_lab < 0:
                    tt_labs.append(str(int(round(theta_lab+360))) + '$^\circ$')
                else:
                    tt_labs.append(str(int(round(theta_lab))) + '$^\circ$')
            ax.set_thetagrids(tt, labels=tt_labs, fontsize=fs)

        ax.grid(linestyle=":", color='black')
 
        if include_cbar:
            fig.colorbar(im,location='right',shrink=.65,label=f"{c} [{info['units']}]")
        if include_title:
            ax.set_title(c)


#-----------------------------------------------------------------------------------------------
def plot_from_wdss2(file_path,
                    varname,
                    rangemax=300,
                    Xlat=None,
                    Xlon=None,
                    patch_size=None,
):
    """
    Plots a PPI from WDSS2-decoded netcdfs.
    Args:
    - file_path (str): full path to the netcdf
    - varname (str or list of str): variable name(s)
    - rangemax (int): the maximum range to plot, in km from radar
    - Xlat (float or None): An optional latitude to plot
    - Xlon (float or None): An optional longitude to plot
    - patch_size tuple (int, int): +/- grid points from Xlat/Xlon grid point in polar coordinates for segment plot 
    """

    if patch_size:
        n_az_patch, n_gate_patch = patch_size
    else:
        n_az_patch, n_gate_patch = None, None

    data = {}

    if isinstance(varname, str):
        varnames = [varname]
        file_paths = [file_path]
        oneplot = True
    else:
        # It is a list of strings. This mean we want a multipanel.
        oneplot = False
        varnames = varname
        tilt = os.path.basename(os.path.dirname(file_path))
        raddt = datetime.strptime(os.path.basename(file_path), '%Y%m%d-%H%M%S.netcdf')
        rootdir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))   
      
        # Find closest files
        file_paths = []
        for vname in varnames:
            all_files = glob.glob(f"{rootdir}/{vname}/{tilt}/*netcdf")
            dts = [datetime.strptime(os.path.basename(ff), '%Y%m%d-%H%M%S.netcdf') for ff in all_files]
            closest_dt = min(dts, key=lambda dt: abs(raddt - dt))
            if abs(raddt - closest_dt).seconds > 180:
                print(f"{raddt} is too far from {closest_dt}")
                sys.exit(1)
            idx = dts.index(closest_dt)
            file_paths.append(all_files[idx])

    if not oneplot:
        # Determine subplot layout based on number of plots
        if len(file_paths) == 2: nrows, ncols, figsize, fs = 1, 2, (12,5), 10
        elif len(file_paths) == 3: nrows, ncols, figsize, fs = 1, 3, (12,5), 10
        elif len(file_paths) == 4: nrows, ncols, figsize, fs = 2, 2, (10,10), 10
        elif len(file_paths) in [5, 6]: nrows, ncols, figsize, fs = 2, 3, (12,8), 8
        elif len(file_paths) in [7, 8]: nrows, ncols , figsize, fs = 2, 4, (12,4), 8
        else:
            print('Too many (or too few) variables to plot!')
            sys.exit(1)

        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)#, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
        axes = axes.flat
        # Removing the initial axes...plot_radar function will configure
        for ax in axes: ax.remove()
    else:
        fig = plt.figure()
        nrows, ncols = None, None


    for ii, file_path in enumerate(file_paths):
        varname = varnames[ii]

        # Open the netCDF file using xarray
        ds = xr.open_dataset(file_path)
        
        raddata = ds[varname].values
        azimuth = ds['Azimuth'].values
        gate_width = ds['GateWidth'].values.mean() # Assuming GateWidth is relatively constant
        range_to_first_gate = ds.attrs['RangeToFirstGate']
        radar_name = ds.attrs['radarName-value'] 

        # Calculate radial distances (r) for each gate
        # The 'Gate' dimension represents the index of the gate, not the actual distance.
        # We need to calculate the distance for each gate based on RangeToFirstGate and GateWidth.
        # The distance to each gate can be calculated as RangeToFirstGate + (gate_index * GateWidth)
        # However, GateWidth is a variable with Azimuth dimension.
        
        # Create an array of gate distances (r)
        gates = np.arange(ds.sizes['Gate'])
        r_meters = range_to_first_gate + (gates * gate_width)
        
        # Convert azimuth from degrees to radians for Matplotlib's polar plot
        theta = np.copy(azimuth)       
        ntheta = len(theta)

        # Handle potential missing data values if needed
        # The ncdump shows MissingData = -99900.f and RangeFolded = -99901.f
        missing_data_value = ds.attrs.get('MissingData', -99900.0)
        range_folded_value = ds.attrs.get('RangeFolded', -99901.0)
        
        # Replace missing values with NaN for plotting
        if varname in ['Reflectivity', 'Zdr', 'RhoHV', 'PhiDP']:
            rad_masked = np.where(raddata == missing_data_value, np.nan, raddata)
        elif varname in ['Velocity', 'AliasedVelocity', 'SpectrumWidth', 'AzShear', 'DivShear']:
            rad_masked = np.where(raddata == missing_data_value, 0, raddata)
            rad_masked = np.where(rad_masked == range_folded_value, np.nan, rad_masked)
      
 
        # Find the index where the range exceeds the limit
        max_range_index = np.where(r_meters > rangemax * 1000)[0] # Convert km to meters for comparison

        if len(max_range_index) > 0:
            # Use the index of the first gate that exceeds the limit
            end_gate_index = max_range_index[0]
            r_meters_limited = r_meters[:end_gate_index]
            rad_masked_limited = rad_masked[:, :end_gate_index]
        else:
            # If all gates are within the limit, use the full data
            r_meters_limited = r_meters
            rad_masked_limited = rad_masked 
  
        # Convert the limited range to kilometers
        r_km_limited = r_meters_limited / 1000.0
   
 
        # Get center location, if desired
        if Xlat is not None and Xlon is not None:
            try:
                azimuth_idx, gate_idx, calc_az, calc_range = get_azimuth_range_from_latlon(Xlat, Xlon, ds=ds)
            except ValueError as err:
                print(err); sys.exit(1)
            
            calc_range /= 1000 # convert to km
            calc_az_rad = math.radians(calc_az)
            
            # Gate slicing (clamped)
            num_gates_limited = len(r_km_limited)
            start_gate_idx = max(0, gate_idx - n_gate_patch)
            end_gate_idx = min(num_gates_limited, gate_idx + n_gate_patch + 1) # +1 for exclusive end
            r_to_plot = r_km_limited[start_gate_idx:end_gate_idx]
            data['rng_lower'] = r_plot_min = r_to_plot.min()
            data['rng_upper'] = r_plot_max = r_to_plot.max()

            # Azimuth slicing (clamped, non-wrapping)
            num_azimuths = len(theta)
            start_az_idx = azimuth_idx - n_az_patch 
            end_az_idx = azimuth_idx + n_az_patch + 1 

            try:
                data['az_upper'] = theta[azimuth_idx + n_az_patch] # in degrees
            except IndexError:
                data['az_upper'] = theta[azimuth_idx + n_az_patch - ntheta]
            data['az_lower'] = theta[azimuth_idx - n_az_patch] 

            if end_az_idx > num_azimuths:
                # Indicates that we are wrapping around in DATA SPACE (not necessarily in azimuth)
                rad_to_plot = np.concatenate([
                    rad_masked_limited[start_az_idx:, start_gate_idx:end_gate_idx],
                    rad_masked_limited[0:(end_az_idx-num_azimuths), start_gate_idx:end_gate_idx]
                ])
            elif start_az_idx < 0:
                # Indicates that we are wrapping around in DATA SPACE (not necessarily in azimuth)
                rad_to_plot = np.concatenate([
                    rad_masked_limited[num_azimuths+start_az_idx:, start_gate_idx:end_gate_idx],
                    rad_masked_limited[:end_az_idx, start_gate_idx:end_gate_idx]
                ])
            else:
                rad_to_plot = rad_masked_limited[start_az_idx:end_az_idx, start_gate_idx:end_gate_idx]

            data[varname] = {'data':rad_to_plot}
            full_ppi = False
        else:       
            # Plot the full limited range
            data[varname] = {'azimuths':theta, 'ranges':r_km_limited}
            data[varname]['data'] = rad_masked_limited
            full_ppi = True

        # Plot an X at the given location
        if Xlat is not None and Xlon is not None:
            #ax.plot(calc_az_rad, calc_range, markersize=15, marker='o', markeredgewidth=3,  markerfacecolor='none', color='black')
            if oneplot: ax.plot(calc_az_rad, calc_range, markersize=15, marker='o', markeredgewidth=3,  markerfacecolor='none', color='black')

        # Set title
        scan_time = ds.attrs['Time']
        dt_object = datetime.utcfromtimestamp(scan_time)
#        if oneplot:
#            ax.set_title(f'{radar_name} {varname} PPI (Elevation: {ds.attrs["Elevation"]} {ds.attrs["ElevationUnits"]})\nTime: {dt_object} UTC', va='bottom')

        # Close the dataset
        elev = ds.attrs["Elevation"]
        elev_units = ds.attrs["ElevationUnits"]
        ds.close()
   
    plot_radar(data,
               channels=varnames,
               fig=fig,
               include_cbar=True,
               include_title=False,
               full_ppi=full_ppi
    )
 
    if not oneplot:
        plt.suptitle(f'{radar_name} PPI (Elevation: {elev} {elev_units})\nTime: {dt_object} UTC', va='bottom', y=0.85)
    
    
    return plt.gcf(), radar_name
#-------------------------------------------------------------------------------------------------------------

# --- Functions for radar geometry (accounting for Earth's curvature) ---
# These equations convert ground range to slant range (and vice-versa)
# based on a simplified Earth model with a constant effective Earth radius.
# For more advanced radar applications, consider libraries like Py-ART
# or wradlib which have robust georeferencing modules.

def ground_range_to_slant_range(ground_range, radar_height_m, elevation_deg, earth_radius_m=6371000):
    """
    Converts ground range to slant range, accounting for Earth's curvature.
    Assumes a standard atmospheric refraction (4/3 Earth radius model usually).
    """
    elevation_rad = np.deg2rad(elevation_deg)
    # The effective Earth radius is often scaled by 4/3 for atmospheric refraction
    # This is a common approximation in radar meteorology
    Re = earth_radius_m * (4/3)

    a = ground_range**2 + (radar_height_m / np.cos(elevation_rad))**2
    b = 2 * ground_range * (radar_height_m / np.cos(elevation_rad))
    slant_range_squared = a + b * np.sin(elevation_rad)
    
    # A more common, simpler formula for slant range given ground range and elevation
    # is often approximated directly from trigonometry after accounting for curvature.
    # Let's use a more direct geometric approach for slant range:
    
    # Height of target above radar level for a given ground range and elevation
    # This formula is for height from radar, accounting for beam path
    target_height_above_radar = ground_range * np.tan(elevation_rad) + \
                                ground_range**2 / (2 * Re * np.cos(elevation_rad)**2)

    # Slant range from Pythagorean theorem, considering target height above radar
    slant_range = np.sqrt(ground_range**2 + (target_height_above_radar + radar_height_m)**2)
    
    # Using simple trigonometry for slant range (hypotenuse) from ground range (adjacent)
    # and vertical displacement (opposite). We need height of beam at ground range.
    # Effective Earth radius
    k = 4/3 # standard atmospheric refraction factor
    Re_eff = earth_radius_m * k

    # Height of beam above radar at ground range (d_ground)
    # This is the height of the beam center relative to the radar.
    h_beam_above_radar = (ground_range * np.tan(elevation_rad)) + \
                         (ground_range**2 / (2 * Re_eff))

    # Slant range is the hypotenuse
    slant_range = np.sqrt(ground_range**2 + h_beam_above_radar**2)
    
    return slant_range

#----------------------------------------------------------------------------------------------------------------------

def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing (azimuth) from point 1 to point 2.
    Returns bearing in degrees, 0-360, where 0 is North, 90 East.
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    initial_bearing_rad = math.atan2(x, y)
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    
    # Ensure bearing is in 0-360 range
    return (initial_bearing_deg + 360) % 360

#--------------------------------------------------------------------------------------------------------------------------

def get_azimuth_range_from_latlon(
    target_lat,
    target_lon,
    ds=None,
    file_path=None,
    print_test=False,
):

    if file_path is not None:
        # Open the netCDF file
        ds = xr.open_dataset(file_path)
    elif ds is None:
        raise ValueError("ds is None")

    # --- 1. Get Radar Location ---
    radar_lat = ds.attrs['Latitude']
    radar_lon = ds.attrs['Longitude']
    radar_height_m = ds.attrs['Height']
    elevation_deg = ds.attrs['Elevation'] # Fixed elevation for this PPI scan

    radar_point = Point(radar_lat, radar_lon)
    target_point = Point(target_lat, target_lon)

    # --- 2. Calculate Distance and Azimuth from Radar to Target ---
    # Geodesic distance (shortest distance on the surface of the ellipsoid)
    # This gives ground distance
    ground_distance_m = geodesic(radar_point, target_point).meters

    # Calculate azimuth (bearing) from radar to target
    calculated_azimuth_deg = calculate_initial_bearing(radar_lat, radar_lon, target_lat, target_lon)

    # --- 3. Account for Earth Curvature (Slant Range) ---
    # Convert ground distance to slant range using the approximate function
    # This step is crucial because the 'Gate' dimension corresponds to slant range measurements.
    calculated_slant_range_m = ground_range_to_slant_range(ground_distance_m, radar_height_m, elevation_deg)

    # --- 4. Find Nearest Azimuth Index ---
    # Get the azimuth values from the dataset
    azimuths_data = ds['Azimuth'].values

    # Find the index of the closest azimuth in the dataset
    azimuth_index = np.argmin(np.abs(azimuths_data - calculated_azimuth_deg))
    closest_azimuth = azimuths_data[azimuth_index]

    # --- 5. Find Nearest Gate Index ---
    # First, reconstruct the range (distance) for each gate in meters, as done for plotting.
    # Use the mean GateWidth for simplicity, or if GateWidth varies per azimuth,
    # you might need to find the appropriate GateWidth for the found azimuth_index.
    # For now, let's use the average.
    gates = ds['GateWidth'].values
    gate_width = gates.mean()
    range_to_first_gate = ds.attrs['RangeToFirstGate']
    
    # Reconstruct the slant ranges corresponding to each gate index
    # (these are the 'r' values used in plotting)
    all_gate_slant_ranges_m = range_to_first_gate + (np.arange(ds.sizes['Gate']) * gate_width)

    # Find the index of the closest gate (range) in the dataset
    gate_index = np.argmin(np.abs(all_gate_slant_ranges_m - calculated_slant_range_m))
    
    if gate_index == (len(all_gate_slant_ranges_m) - 1):
        raise ValueError("Lat/lon not within this volume")

    closest_gate_range = all_gate_slant_ranges_m[gate_index]

    if print_test:
        print(f"--- Radar Location ---")
        print(f"Latitude: {radar_lat}, Longitude: {radar_lon}, Height: {radar_height_m} m")
        print(f"Scan Elevation: {elevation_deg} degrees")
        print(f"\n--- Target Location ---")
        print(f"Target Latitude: {target_lat}, Target Longitude: {target_lon}")
    
        print(f"\n--- Calculated Values to Target ---")
        print(f"Ground Distance: {ground_distance_m:.2f} meters")
        print(f"Calculated Azimuth (from Radar to Target): {calculated_azimuth_deg:.2f} degrees")
        print(f"Estimated Slant Range (accounting for curvature): {calculated_slant_range_m:.2f} meters")
    
        print(f"\n--- Found Indices ---")
        print(f"Nearest Azimuth Index: {azimuth_index} (Actual Azimuth in data: {closest_azimuth:.2f} degrees)")
        print(f"Nearest Gate Index: {gate_index} (Actual Slant Range in data: {closest_gate_range:.2f} meters)")
    
        # You can now access the reflectivity value at these indices:
        reflectivity_at_point = ds['Reflectivity'].values[azimuth_index, gate_index]
        print(f"Reflectivity at target location: {reflectivity_at_point:.2f} {ds['Reflectivity'].units}")

    # Close the dataset, if we opened it
    if file_path is not None:
        ds.close()

    return azimuth_index, gate_index, calculated_azimuth_deg, calculated_slant_range_m


if __name__ == "__main__":
    
    #file_path = '/data/thea.sandmael/data/radar/20160509/KLSX/netcdf/Reflectivity/00.50/20160509-221256.netcdf'

    # Mayfield, KY 12/11/21
    #file_path = '/work/thea.sandmael/radar/20211211/KPAH/netcdf/AzShear/00.50/20211211-032407.netcdf'
    #file_path = '/work/thea.sandmael/radar/20211211/KPAH/netcdf/Reflectivity/00.50/20211211-032544.netcdf'
    #target_lat, target_lon = 36.74, -88.64

    # North of radar example
    #file_path = '/data/thea.sandmael/data/radar/20120220/KVNX/netcdf/Velocity/00.50/20120220-211430.netcdf'
    #file_path = '/data/thea.sandmael/data/radar/20120220/KVNX/netcdf/Reflectivity/00.50/20120220-211412.netcdf'
    #target_lat, target_lon = 37.29, -98.03
    
    # 127 km away example
    #file_path = '/myrorss2/data/thea.sandmael/data/radar/20140618/KFSD/netcdf/Velocity/00.50/20140618-030239.netcdf'
    #target_lat, target_lon = 42.4988, -97.0407

    # 166 km away example
    #file_path = '/myrorss2/data/thea.sandmael/data/radar/20151223/KNQA/netcdf/Velocity/00.50/20151223-205557.netcdf'
    #target_lat, target_lon = 34, -90.71

    # Wisconsin example
    #file_path = '/myrorss2/work/thea.sandmael/radar/20240522/KARX/netcdf/Velocity/00.50/20240522-002936.netcdf'
    #target_lat, target_lon = 44.75, -90.5

    # Greenfield, Iowa
    #file_path = '/myrorss2/work/thea.sandmael/radar/20240521/KDMX/netcdf/Velocity/00.50/20240521-204123.netcdf'
    #target_lat, target_lon = 41.3, -94.51

    # 9-10 km away example
    #file_path = '/myrorss2/data/thea.sandmael/data/radar/20160524/KDDC/netcdf/Velocity/00.50/20160524-235541.netcdf'
    #target_lat, target_lon = 37.77, -99.99 #37.7922, -100.0695

    # Due north example
    #file_path = '/myrorss2/work/thea.sandmael/radar/20240529/KFDX/netcdf/Velocity/00.50/20240529-003956.netcdf'
    #target_lat, target_lon = 35.5278, -103.543

    # Crosses due North
    file_path = '/myrorss2/work/thea.sandmael/radar/20240613/KLSX/netcdf/Velocity/00.50/20240613-223435.netcdf'
    target_lat, target_lon = 40.0899, -91.7384

    # EF4 example (KFWS)
    #file_path = '/myrorss2/data/thea.sandmael/data/radar/20170429/KFWS/netcdf/Velocity/00.50/20170429-230006.netcdf' #20170429-230946.netcdf'
    #target_lat, target_lon = 32.51, -95.91

    patch_size = 64, 128 #120//2, 240//2 # n_az_patch, n_gate_patch # these are half-sizes

    #varname = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    varname = ['Reflectivity', 'Velocity'] #, 'RhoHV', 'AzShear'] #'RhoHV', 'Zdr', 'PhiDP', 'Velocity', 'SpectrumWidth', 'AzShear', 'DivShear']

    fig, radar = plot_from_wdss2(file_path,
                                 varname=varname,
                                 Xlat=target_lat,
                                 Xlon=target_lon,
                                 patch_size=patch_size,
                                 rangemax=300,
    )

    if isinstance(varname, str):
        figname = f'{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')
    #print_rad_val_at_latlon(file_path, target_lat, target_lon)
