# torcnn
Using the database created by Sandmæl et al. (2023), we explore tornado predictions with CNNs.

## Recipe for environment for using GPU

```
# Create and activate the environment
conda create --name tf-gpu python=3.10
conda activate tf-gpu

# Install tensorflow w/ cuda
pip install tensorflow[and-cuda]

# Use python to check that GPUs are recognizable
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install other necessary packages
pip install xarray numpy pandas matplotlib netCDF4 numba
```
