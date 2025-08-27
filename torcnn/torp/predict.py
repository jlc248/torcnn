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

modelfile = '/raid/jcintineo/torcnn/torp_datasets/new_rf01/random_forest_pipeline.joblib'

# torp v1
#modelfile = '/raid/jcintineo/torcnn/torp_datasets/NTDArandomForest_6047_falsealarms.pkl'

outdir = '/raid/jcintineo/torcnn/eval/nontor2024_pretor2013/all/new_rf01'
os.makedirs(outdir, exist_ok=True)

# Read the data from the CSV file
#df = pd.read_csv('/raid/jcintineo/torcnn/torp_datasets/2023_Storm_Reports_Expanded_tilt0050_radar_r2500_nodup.csv')
csvfile = glob.glob(f"{os.path.dirname(outdir)}/*csv")
if len(csvfile) == 1:
    csvfile = csvfile[0]
else:
    print(f'ambiguous csvs: {csvfile}')
    sys.exit(1)

df = pd.read_csv(csvfile)

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
    labels = ((df['tornado'] == 1) | (df['preTornado'] == 1)).astype(int)
except KeyError:
    labels = (df['tornado'] == 1).astype(int)

print('fraction of positive:', np.round(np.sum(labels)/len(labels),4))
np.save(f'{outdir}/labels.npy', labels)

# Add the predictions as a new column to the DataFrame
#df['prediction'] = predictions

# Save the DataFrame with predictions to a new CSV file
#df.to_csv('predictions.csv', index=False)
