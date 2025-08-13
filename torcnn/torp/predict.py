import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

# Use conda environment "old_sklearn"!

outdir = '/raid/jcintineo/torcnn/torp_datasets/eval2023/no_spout/'
os.makedirs(outdir, exist_ok=True)

# Read the data from the CSV file
#df = pd.read_csv('/raid/jcintineo/torcnn/torp_datasets/2023_Storm_Reports_Expanded_tilt0050_radar_r2500_nodup.csv')
df = pd.read_csv('/raid/jcintineo/torcnn/eval/nospout_2023/torp_2023_nospout_cleaned.csv')

# Load the list of feature names from the pickle file
with open('torp_features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load the trained RandomForestClassifier model from the pickle file
with open('/raid/jcintineo/torcnn/torp_datasets/NTDArandomForest_6047_falsealarms.pkl', 'rb') as f:
    model = pickle.load(f)

# Sub-sample, if necessary
df = df[df.spout == 0]

# Select the feature columns from the DataFrame
X = df[feature_names]

# Make predictions using the loaded model
predictions = model.predict_proba(X)
np.save(f'{outdir}/predictions.npy', predictions[:,1])

labels = np.array(df['tornado'])
np.save(f'{outdir}/labels.npy', labels)

# Add the predictions as a new column to the DataFrame
#df['prediction'] = predictions

# Save the DataFrame with predictions to a new CSV file
#df.to_csv('predictions.csv', index=False)
