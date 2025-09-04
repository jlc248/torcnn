import keras
import sys
sys.path.append('../')
import time
from dealias import VelocityDealiaser
import rad_utils
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

t0=time.time()
vda = keras.models.load_model('dealias_sn16_csi9764.keras',compile=False)
print('time to read model:',time.time()-t0)

file_path = '/myrorss2/work/thea.sandmael/radar/20240613/KLSX/netcdf/AliasedVelocity/00.50/20240613-223435.netcdf'
ds = xr.open_dataset(file_path)
raddata = ds['AliasedVelocity'].values
raddata = raddata[:,0:1152] # making it compatible with the model
azimuth = ds['Azimuth'].values
gate_width = ds['GateWidth'].values.mean() 
range_to_first_gate = ds.attrs['RangeToFirstGate']
radar_name = ds.attrs['radarName-value']
nyq = np.array(np.float32(ds.attrs['Nyquist_Vel-value'])).reshape(1,)
raddata = np.expand_dims(raddata, axis=(0,-1))

n_frames=1

# Add batch dim
raddata=raddata[None,...]
nyq=nyq[None,...]

# Split 0.5 degree data into two 1 degree arrays
s = raddata.shape
l2 = np.reshape(raddata,(s[0],s[1],s[2]//2,2,s[3],s[4]))
l2 = np.transpose(l2,(0,3,1,2,4,5))
l2 = np.reshape(l2, (2*s[0],s[1],s[2]//2,s[3],s[4]))
nyq=np.stack((nyq,nyq))
nyq=np.transpose(nyq,(1,0,2))
nyq=np.reshape(nyq,(-1,n_frames,1))

# Pad data 12 degrees on either side with periodic boundary conditions
pad_deg=12
l2 = np.concatenate( (l2[:,:,-pad_deg:,:,:],
                                l2,
                                l2[:,:,:pad_deg,:,:]),axis=2)  

# Replace bad values with NAN
l2[l2<=-64]=np.nan

print(l2.shape, nyq.shape)

# Run UNet
inp = {'vel':l2,'nyq':nyq}
t0=time.time()
out=vda.predict(inp)
print('predict time',time.time()-t0)
dealiased_vel=out['dealiased_vel'].copy()

## Post process
# Recombine to 0.5 degree 
s=dealiased_vel.shape
dealiased_vel = np.reshape(dealiased_vel,(s[0]//2,2,s[1],s[2],s[3]))
dealiased_vel = np.transpose(dealiased_vel,(0,2,1,3,4))
dealiased_vel = np.reshape(dealiased_vel,(s[0]//2,-1,s[2],s[3]))

# remove padding
dealiased_vel = dealiased_vel[:, 24:-24, :, :]
print(dealiased_vel.shape)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].imshow(dealiased_vel[0,:,:,0], cmap='PiYG', vmin=-50, vmax=50)
# Fix bad vals
raddata[raddata <=-64]=np.nan
ax[1].imshow(np.squeeze(raddata), cmap='PiYG', vmin=-50, vmax=50)
plt.show()
