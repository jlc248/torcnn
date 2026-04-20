import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
import glob
import sys
import joblib

# Use conda environment "old_sklearn" for torp v1!
# Also, check labels below. Use 'preTornado == 1' if using pretornado dataset

#modelfile = '/raid/jcintineo/torcnn/torp_datasets/new_rf01/random_forest_pipeline.joblib'

modelfile = '/work2/jcintineo/torcnn/torp_rf/combined_val2018_train2011-19/random_forest_pipeline.joblib'

# torp v1
#modelfile = '/raid/jcintineo/torcnn/torp_datasets/NTDArandomForest_6047_falsealarms.pkl'

outdir = os.path.dirname(modelfile) + '/val/'
os.makedirs(outdir, exist_ok=True)
 
# Read the data from the CSV file
df = pd.read_csv('/work2/jcintineo/TORP/combined_torp_rep_pretor/matches/val_test04.csv')

# Load the trained RandomForestClassifier model from the joblib or pickle file
if modelfile.endswith('joblib'):
    model = joblib.load(modelfile)
    feature_names = model.feature_names_in_
else:
    # if older v1, load the list of feature names from the pickle file
    with open('torp_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    with open(modelfile, 'rb') as f:
        model = pickle.load(f)


# Sub-sample, if necessary
try:
    df = df[df.spout == 0]
except AttributeError as err:
    print(str(err))
    print('continuing...')

# Select the feature columns from the DataFrame
X = df[feature_names]

# Make predictions using the loaded model
predictions = model.predict_proba(X)
np.save(f'{outdir}/predictions.npy', predictions[:,1])

try:
    labels = ((df['tornado'] == 1) | (df['pretor'] == 1)).astype(int)
except KeyError:
    labels = (df['tornado'] == 1).astype(int)

print('fraction of positive:', np.round(np.sum(labels)/len(labels),4))
np.save(f'{outdir}/labels.npy', labels)


# Write out pickle file for easy plotting
results = {}

#  Iterate through 5% increments (0.05 to 0.95)
thresholds = np.arange(0.05, 1.00, 0.05)

predictions = predictions[:,1]

for t in thresholds:
    # Convert threshold to a clean string for the key (e.g., 0.05 -> '05')
    t_str = f"{int(round(t * 100)):02d}"
    
    # Binarize predictions based on current threshold
    binary_preds = (predictions >= t).astype(int)

    # Calculate components
    tp = np.sum((binary_preds == 1) & (labels == 1))
    fp = np.sum((binary_preds == 1) & (labels == 0))
    fn = np.sum((binary_preds == 0) & (labels == 1))
    
    # Calculate metrics (with guards against division by zero)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Store in dictionary with your specific naming convention
    results[f'pod{t_str}_index0'] = pod
    results[f'far{t_str}_index0'] = far
    results[f'csi{t_str}_index0'] = csi

# --- Part 2: Reliability Diagram Metrics (Binned Counts) ---
bins = np.arange(0.0, 1, 0.05)

for i in range(len(bins) - 1):
    low = bins[i]
    high = bins[i+1]
    
    # Format key 
    high_str = f"{int(round(high * 100)):02d}"
    key_suffix = f"{high_str}_index0"
    
    # Define the mask for the current bin
    # We use [low, high) 
    mask = (predictions >= low) & (predictions < high)
    
    # fcstct: Number of forecasts in this probability bin
    fcst_count = np.sum(mask)
    
    # obsct: Number of positive observations (labels == 1) in this bin
    obs_count = np.sum(labels[mask] == 1)
    
    results[f'fcstct{key_suffix}'] = int(fcst_count)
    results[f'obsct{key_suffix}'] = int(obs_count)


# Save to pkl file
with open(f'{outdir}/eval_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"Successfully saved {len(results)} metrics to {outdir}/eval_results.pkl")
