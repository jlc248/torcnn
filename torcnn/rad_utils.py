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

def plot_ppi(file_path,
             varname,
             rangemax=300,
             Xlat=None,
             Xlon=None,
             window_size=None,
             plot_segment=False
):
    """
    Plots a PPI from WDSS2-decoded netcdfs.
    Args:
    - file_path (str): full path to the netcdf
    - varname (str or list of str): variable name(s)
    - rangemax (int): the maximum range to plot, in km from radar
    - Xlat (float or None): An optional latitude to plot
    - Xlon (float or None): An optional longitude to plot
    - window_size (int): +/- points from Xlat/Xlon in polar coordinates for segment plot 
    - plot_segment (bool): Plot only the segment defined by Xlat, Xlon, and window_size
    """

    if isinstance(varname, str):
        varnames = [varname]
        file_paths = [file_path]
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
        axes = [ax]
        oneplot = True
        fs = 14
    else:
        # It is a list of strings. This mean we want a multipanel.
        oneplot = False
        varnames = varname

        # e.g. file_path
        #/data/thea.sandmael/data/radar/20120220/KVNX/netcdf/Reflectivity/00.50/20120220-211412.netcdf

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

        # Create the polar plot
        if len(file_paths) == 2:
            nrows,ncols = 1,2
            figsize = (12,5)
            fs = 10
        elif len(file_paths) == 3:
            nrows,ncols = 1,3
            figsize = (12,5)
            fs = 10
        elif len(file_paths) == 4:
            nrows,ncols = 2,2
            figsize = (10,10)
            fs = 10
        elif len(file_paths) == 5 or len(file_paths) == 6:
            nrows,ncols = 2,3
            figsize = (12,8)
            fs = 8
        elif len(file_paths) == 7 or len(file_paths) == 8:
            nrows,ncols = 2,4
            figsize = (12,4)
            fs = 8
        else:
            print('Too many variables to plot!')
            sys.exit(1)

        fig, axes = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize, nrows=nrows, ncols=ncols)#, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
        axes = axes.flat

    for ii, file_path in enumerate(file_paths):
        ax = axes[ii]
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
        # Let's assume the gate values are 0-indexed.
        # The distance to each gate can be calculated as RangeToFirstGate + (gate_index * GateWidth)
        # However, GateWidth is a variable with Azimuth dimension. For a simple plot,
        # we can take the mean GateWidth, or if more precision is needed, handle
        # the varying gate width per azimuth. For now, let's use the mean.
        
        # Create an array of gate distances (r)
        gates = np.arange(ds.sizes['Gate'])
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
 
        # Plot the radar data
        if varname == 'Reflectivity':
            cmap = NWScmap; vmin=-10; vmax=75
        elif varname == 'AzShear':
            cmap = 'bwr'; vmin=-0.02; vmax=0.02
        elif varname == 'DivShear':
            cmap = 'PiYG'; vmin=-0.02; vmax=0.02
        elif varname == 'Velocity' or varname == 'AliasedVelocity':
            cmap = 'PiYG'; vmin=-50; vmax=50
        elif varname == 'SpectrumWidth':
            cmap = 'cubehelix'; vmin=0; vmax=10
        elif varname == 'Zdr':
            cmap = 'Spectral_r'; vmin=-5; vmax=5
        elif varname == 'RhoHV':
            cmap = 'rainbow'; vmin=0.45; vmax=1
        elif varname == 'PhiDP':
            cmap = 'gnuplot2'; vmin=0; vmax=360;

        # Get center location, if desired
        if Xlat is not None and Xlon is not None:
            try:
                azimuth_idx, gate_idx, calc_az, calc_range = get_azimuth_range_from_latlon(Xlat, Xlon, ds=ds)
            except ValueError as err:
                print(err); sys.exit(1)
            
            calc_range /= 1000 # convert to km
            calc_az_rad = math.radians(calc_az)
            
            if plot_segment:
                # Azimuth slicing (clamped, non-wrapping)
                num_azimuths = len(theta)
                start_az_slice = azimuth_idx - window_size #max(0, azimuth_idx - window_size)
                end_az_slice = azimuth_idx + window_size + 1 #min(num_azimuths, azimuth_idx + window_size + 1) # +1 for exclusive end 
                    
                # Gate slicing (clamped)
                num_gates_limited = len(r_km_limited)
                start_gate_slice = max(0, gate_idx - window_size)
                end_gate_slice = min(num_gates_limited, gate_idx + window_size + 1) # +1 for exclusive end

                if end_az_slice > num_azimuths:
                    # Indicates that we're wrapping around 0 degrees
                    rad_to_plot = np.concatenate([
                                    rad_masked_limited[start_az_slice:num_azimuths, start_gate_slice:end_gate_slice],
                                    rad_masked_limited[0:(end_az_slice-num_azimuths), start_gate_slice:end_gate_slice]
                    ])
                    theta_to_plot = np.concatenate([
                                      theta[start_az_slice:num_azimuths],
                                      theta[0:(end_az_slice-num_azimuths)]
                    ])

                    # Set az plot limits for the segment
                    # For ax.set_thetamin and ax.set_thetamax, we need to < 0 and > 0 points
                    theta_plot_min_deg = np.degrees(theta_to_plot[0])
                    theta_plot_max_deg = np.degrees(theta_to_plot[-1])
                elif start_az_slice < 0:
                    # Indicates that we're wrapping around 0 degrees
                    rad_to_plot = np.concatenate([
                                    rad_masked_limited[start_az_slice:, start_gate_slice:end_gate_slice],
                                    rad_masked_limited[0:end_az_slice, start_gate_slice:end_gate_slice]
                    ])
                    theta_to_plot = np.concatenate([
                                      theta[start_az_slice:],
                                      theta[0:end_az_slice]
                    ])

                    # Set az plot limits for the segment
                    theta_plot_min_deg = np.degrees(theta_to_plot[0])
                    theta_plot_max_deg = np.degrees(theta_to_plot[-1])
                else:    
                    rad_to_plot = rad_masked_limited[start_az_slice:end_az_slice, start_gate_slice:end_gate_slice]
                    theta_to_plot = theta[start_az_slice:end_az_slice]
                    
                    # Set az plot limits for the segment
                    theta_plot_min_deg = np.degrees(theta_to_plot.min())
                    theta_plot_max_deg = np.degrees(theta_to_plot.max())

                # Range should be unaffected by azimuth wrapping
                r_to_plot = r_km_limited[start_gate_slice:end_gate_slice]
                r_plot_min = r_to_plot.min()
                r_plot_max = r_to_plot.max()
            else:
                # Plot the full limited range
                rad_to_plot = rad_masked_limited
                theta_to_plot = theta
                r_to_plot = r_km_limited

                # Set plot limits for the full PPI
                r_plot_min = 0
                r_plot_max = rangemax
                theta_plot_min_deg = 0
                theta_plot_max_deg = 360 # For setting x-tick labels 

        else:       
            # Plot the full limited range
            rad_to_plot = rad_masked_limited
            theta_to_plot = theta
            r_to_plot = r_km_limited

            # Set plot limits for the full PPI
            r_plot_min = 0
            r_plot_max = rangemax
            theta_plot_min_deg = 0
            theta_plot_max_deg = 360 # For setting x-tick labels

        # Use pcolormesh for a 2D color plot in polar coordinates
        # theta should be 2D, r should be 2D for pcolormesh to work correctly
        # We need to broadcast theta and r to match the shape of raddata
        R, THETA = np.meshgrid(r_to_plot, theta_to_plot)
        
        c = ax.pcolormesh(THETA, R, rad_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if plot_segment:
            # Hide gridlines, axis border, and tick labels
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
 
        # Plot an X at the given location
        if Xlat is not None and Xlon is not None:
            if oneplot: ax.plot(calc_az_rad, calc_range, markersize=15, marker='o', markeredgewidth=3,  markerfacecolor='none', color='black')
  
        # Add a colorbar
        cbar = fig.colorbar(c, ax=ax, orientation='vertical', pad=0.1, shrink=0.75)
        if ds[varname].units == 'MetersPerSecond':
            label = 'm/s'
        elif ds[varname].units == 'dimensionless':
            label = ' '
        else: 
            label = ds[varname].units
        cbar.set_label(f'{varname} ({label})', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs-2)       
 
        # Set plot properties
        ax.set_theta_zero_location('N')  # North at the top
        ax.set_theta_direction(-1)      # Clockwise direction (standard for radar)
       
        # Customize radial ticks (range)
        ax.set_rlabel_position(90) # Position of the radial labels
        if oneplot:
            ax.set_ylabel(f'Range (km)', labelpad=30) # Label for radial axis
        
        # Set custom radial ticks for kilometers
        if plot_segment:
            # Create a few ticks within the zoomed range
            r_ticks = np.round(np.linspace(r_plot_min, r_plot_max, num=4, endpoint=True)).astype(int)
        else:
            r_ticks = np.arange(0, rangemax + 1, 50)

        # Customize angular ticks (azimuth) based on segment or full plot
        if plot_segment:

            # Set the sector
            ax.set_thetamin(theta_plot_min_deg)
            ax.set_thetamax(theta_plot_max_deg)
           
            # Dynamically calculate angular ticks for the segment
            # Ensure degrees are within 0-360 range for display
            start_az_deg = np.degrees(theta_to_plot.min()) % 360
            end_az_deg = np.degrees(theta_to_plot.max()) % 360

            # Rounded to nearest 5 or 10 degrees for readability
            num_desired_ticks = 4
            approx_interval = (end_az_deg - start_az_deg) / (num_desired_ticks - 1)
            # Round interval to nearest nice number (e.g., 1, 2, 5, 10)
            if approx_interval > 10:
                tick_interval = round(approx_interval / 10) * 10
            elif approx_interval > 5:
                tick_interval = 5
            elif approx_interval > 2:
                tick_interval = 2
            else:
                tick_interval = 1
            tick_interval = max(1, tick_interval) # Ensure it's at least 1

            if abs(end_az_deg - start_az_deg) > 180:
                az_fold = True
                # Indicates that we cross the azimuth folding line (north)
                left_flank = np.degrees(theta_to_plot[0])
                right_flank = np.degrees(theta_to_plot[-1])
                
                # Transform these flanks so that they don't cross the folding line
                left_flank_trans = 90 - (360 - left_flank) # assumes left flank is always in Q4
                right_flank_trans = right_flank + 90
                
                # New interval
                approx_interval = (right_flank_trans - left_flank_trans) / (num_desired_ticks - 1)
                tick_interval = round(approx_interval / 10) * 10
                transformed_ticks = np.arange(left_flank_trans, right_flank_trans, tick_interval)
                
                # Transform back to North being 0 degrees
                azimuth_ticks = transformed_ticks - 90
                azimuth_ticks[azimuth_ticks < 0] = 360 + azimuth_ticks[azimuth_ticks < 0]
            else:
                az_fold = False
                azimuth_ticks = np.round(np.arange(start_az_deg, end_az_deg, tick_interval), 1)

            azimuth_ticks_rad = np.deg2rad(azimuth_ticks)
            
            # Plot custom azimuth labels (lines and text)
            for angle, label in zip(azimuth_ticks_rad, azimuth_ticks):
                ax.plot([angle, angle], [r_plot_min, r_plot_max], color='black', linestyle=':', linewidth=0.8)
                ax.text(angle, r_plot_max + 0.1, f'{int(round(label))}°', ha='center', va='bottom', fontsize=fs-2)
            
            # Plot custom range labels (arcs and text)
            for r_val in r_ticks:
                if az_fold:
                    # Divide the line plotting over 0 degrees
                    angles_for_label = np.linspace(azimuth_ticks_rad[0], np.deg2rad(360), 50)
                    ax.plot(angles_for_label, np.full_like(angles_for_label, r_val), color='black', linestyle=':', linewidth=0.8)
                    ax.text(theta_to_plot.max() + np.deg2rad(5), r_val, str(r_val), ha='left', va='center', fontsize=fs-2)
                   
                    angles_for_label = np.linspace(0, azimuth_ticks_rad[-1], 50)
                    ax.plot(angles_for_label, np.full_like(angles_for_label, r_val), color='black', linestyle=':', linewidth=0.8)
                    ax.text(theta_to_plot.max() + np.deg2rad(5), r_val, str(r_val), ha='left', va='center', fontsize=fs-2)
                else:
                    angles_for_label = np.linspace(theta_to_plot.min(), theta_to_plot.max(), 50)
                    ax.plot(angles_for_label, np.full_like(angles_for_label, r_val), color='black', linestyle=':', linewidth=0.8)
                    ax.text(theta_to_plot.max() + np.deg2rad(5), r_val, str(r_val), ha='left', va='center', fontsize=fs-2)
 
            #ax.set_xticks(np.deg2rad(segment_azimuth_ticks))
            #ax.set_xticklabels([f'{a:.0f}°' for a in segment_azimuth_ticks])

            # Adjust radial limit to make space for labels 
            #ax.set_rmax(r_plot_max + 0.2) 

        else:
            ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
            ax.set_xticklabels([f'{a}°' for a in np.arange(0, 360, 30)])
        
        # Set title
        scan_time = ds.attrs['Time']
        dt_object = datetime.utcfromtimestamp(scan_time)
        if oneplot:
            ax.set_title(f'{radar_name} {varname} PPI (Elevation: {ds.attrs["Elevation"]} {ds.attrs["ElevationUnits"]})\nTime: {dt_object} UTC', va='bottom')
    
    if not oneplot:
        plt.suptitle(f'{radar_name} PPI (Elevation: {ds.attrs["Elevation"]} {ds.attrs["ElevationUnits"]})\nTime: {dt_object} UTC', va='bottom', y=0.85)
    
    # Close the dataset
    ds.close()
    
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
    file_path = '/data/thea.sandmael/data/radar/20120220/KVNX/netcdf/Reflectivity/00.50/20120220-211412.netcdf'
    target_lat, target_lon = 37.29, -98.03
    
    # 127 km away example
    #file_path = '/data/thea.sandmael/data/radar/20140618/KFSD/netcdf/Velocity/00.50/20140618-030239.netcdf'
    #target_lat, target_lon = 42.4988, -97.0407

    # 166 km away example
    #file_path = '/data/thea.sandmael/data/radar/20151223/KNQA/netcdf/Velocity/00.50/20151223-205557.netcdf'
    #target_lat, target_lon = 34, -90.71

    # Wisconsin example
    #file_path = '/work/thea.sandmael/radar/20240522/KARX/netcdf/Velocity/00.50/20240522-002936.netcdf'
    #target_lat, target_lon = 44.75, -90.5

    # Greenfield, Iowa
    #file_path = '/work/thea.sandmael/radar/20240521/KDMX/netcdf/Velocity/00.50/20240521-204123.netcdf'
    #target_lat, target_lon = 41.3, -94.51

    # 9-10 km away example
    #file_path = '/data/thea.sandmael/data/radar/20160524/KDDC/netcdf/Velocity/00.50/20160524-235541.netcdf'
    #target_lat, target_lon = 37.7922, -100.0695

    # EF4 example (KFWS)
    file_path = '/data/thea.sandmael/data/radar/20170429/KFWS/netcdf/Velocity/00.50/20170429-230946.netcdf'
    target_lat, target_lon = 32.55, -95.93

    window_size = 96

    #varname = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    varname = ['Reflectivity', 'Velocity', 'RhoHV', 'AzShear'] #'RhoHV', 'Zdr', 'PhiDP', 'Velocity', 'SpectrumWidth', 'AzShear', 'DivShear']

    fig, radar = plot_ppi(file_path,
                          varname=varname,
                          Xlat=target_lat,
                          Xlon=target_lon,
                          window_size=window_size,
                          rangemax=300,
                          plot_segment=True,
    )

    if isinstance(varname, str):
        figname = f'{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')
    #print_rad_val_at_latlon(file_path, target_lat, target_lon)
