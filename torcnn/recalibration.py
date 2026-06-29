import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, precision_recall_curve
import pandas as pd
import os
import joblib

# Load data 
eval_data = '/work2/jcintineo/torcnn/tests/2011-19/test07/eval2018/eval_data.csv'
df = pd.read_csv(eval_data)

y_true = np.array(df.tornado)
y_prob = np.array(df.prob)

isotonic = True 
spline = False
# If both are False, 
# then Platt scaling is used

if isotonic:
    # Fit the Isotonic Regression
    # 'out_of_bounds=clip' ensures that if the test set has a value 
    # slightly higher than the val set, it doesn't crash.
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_prob, y_true)
    recal_type = 'isotonic'

    # Transform probs
    y_prob_calibrated = calibrator.transform(y_prob)
elif spline:
    # Spline fit
    # Setup the Calibrator Pipeline
    # n_knots=5 is usually the "Goldilocks" zone: flexible but not twitchy.
    # degree=3 ensures smooth cubic curves.
    recal_type = 'spline'
    calibrator = make_pipeline(
        SplineTransformer(n_knots=3, degree=3, extrapolation="linear"),
        LogisticRegression(penalty=None) # We want the raw fit to the spline
    )
   
    # Fit on validation data
    calibrator.fit(y_prob.reshape(-1, 1), y_true)
    
    # Transform probs
    y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
else:
    recal_type = 'platt'
    # Initialize the Platt Scaler (Logistic Regression)
    # We set solver='lbfgs' and penalty=None because we want a 
    # pure fit to the calibration curve without regularization.
    calibrator = LogisticRegression(penalty=None, solver='lbfgs')
    
    # Fit on validation Data
    calibrator.fit(y_prob.reshape(-1, 1), y_true)
    
    # Transform probs
    y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    
# Save the calibrator
outdir = os.path.dirname(os.path.dirname(eval_data))
joblib.dump(calibrator, f'{outdir}/tornado_calibrator_{recal_type}.joblib')

# Verification
brier_before = brier_score_loss(y_true, y_prob)
brier_after = brier_score_loss(y_true, y_prob_calibrated)

print(f"Brier Score Before: {brier_before:.5f}")
print(f"Brier Score After:  {brier_after:.5f}")

# Find the new Optimal Threshold for CSI
precision_orig, recall_orig, thresholds_orig = precision_recall_curve(y_true, y_prob)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob_calibrated)
# Assuming CSI formula: CSI = 1 / (1/prec + 1/rec - 1)
csi_orig = 1 / (1/precision_orig + 1/recall_orig - 1 + 1e-10)
ix_orig = np.argmax(csi_orig)
csi = 1 / (1/precision + 1/recall - 1 + 1e-10)
ix = np.argmax(csi)
print(f"ALL Max CSI before: {csi_orig[ix_orig]:.4f} at Threshold: {thresholds_orig[ix_orig]:.4f}")
print(f"ALL Max CSI after: {csi[ix]:.4f} at Threshold: {thresholds[ix]:.4f}")

# For notor vs pretor only
df2 = df[(df.pretorMinutes > 0) | (df.tornado == 0)]
y_true = np.array(df2.tornado)
y_prob = np.array(df2.prob)
if isotonic:
    y_prob_calibrated = calibrator.transform(y_prob)
else:
    y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
precision_orig, recall_orig, thresholds_orig = precision_recall_curve(y_true, y_prob)
csi_orig = 1 / (1/precision_orig + 1/recall_orig - 1 + 1e-10)
ix_orig = np.argmax(csi_orig)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob_calibrated)
csi = 1 / (1/precision + 1/recall - 1 + 1e-10)
ix = np.argmax(csi)
print(f"Pretor vs. nontor Max CSI before: {csi_orig[ix_orig]:.4f} at Threshold: {thresholds_orig[ix_orig]:.4f}")
print(f"Pretor vs. nontor Max CSI after: {csi[ix]:.4f} at Threshold: {thresholds[ix]:.4f}\n")


# For notor vs 1-10 min pretor only
df2 = df[((df.pretorMinutes > 0) & (df.pretorMinutes <= 10)) | (df.tornado == 0)]
y_true = np.array(df2.tornado)
y_prob = np.array(df2.prob)
if isotonic:
    y_prob_calibrated = calibrator.transform(y_prob)
else:
    y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
precision_orig, recall_orig, thresholds_orig = precision_recall_curve(y_true, y_prob)
csi_orig = 1 / (1/precision_orig + 1/recall_orig - 1 + 1e-10)
ix_orig = np.argmax(csi_orig)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob_calibrated)
csi = 1 / (1/precision + 1/recall - 1 + 1e-10)
ix = np.argmax(csi)
print(f"Pretor (1-10 min) vs. nontor Max CSI before: {csi_orig[ix_orig]:.4f} at Threshold: {thresholds_orig[ix_orig]:.4f}")
print(f"Pretor (1-10 min) vs. nontor Max CSI after: {csi[ix]:.4f} at Threshold: {thresholds[ix]:.4f}\n")


# For notor vs 11-20 min pretor only
df2 = df[((df.pretorMinutes > 10) & (df.pretorMinutes <= 20)) | (df.tornado == 0)]
y_true = np.array(df2.tornado)
y_prob = np.array(df2.prob)
if isotonic:
    y_prob_calibrated = calibrator.transform(y_prob)
else:
    y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
precision_orig, recall_orig, thresholds_orig = precision_recall_curve(y_true, y_prob)
csi_orig = 1 / (1/precision_orig + 1/recall_orig - 1 + 1e-10)
ix_orig = np.argmax(csi_orig)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob_calibrated)
csi = 1 / (1/precision + 1/recall - 1 + 1e-10)
ix = np.argmax(csi)
print(f"Pretor (11-20 min) vs. nontor Max CSI before: {csi_orig[ix_orig]:.4f} at Threshold: {thresholds_orig[ix_orig]:.4f}")
print(f"Pretor (11-20 min) vs. nontor Max CSI after: {csi[ix]:.4f} at Threshold: {thresholds[ix]:.4f}\n")


# For notor vs 21-30 min pretor only
df2 = df[((df.pretorMinutes > 20) & (df.pretorMinutes <= 30)) | (df.tornado == 0)]
y_true = np.array(df2.tornado)
y_prob = np.array(df2.prob)
if isotonic:
    y_prob_calibrated = calibrator.transform(y_prob)
else:
    y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
precision_orig, recall_orig, thresholds_orig = precision_recall_curve(y_true, y_prob)
csi_orig = 1 / (1/precision_orig + 1/recall_orig - 1 + 1e-10)
ix_orig = np.argmax(csi_orig)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob_calibrated)
csi = 1 / (1/precision + 1/recall - 1 + 1e-10)
ix = np.argmax(csi)
print(f"Pretor (21-30 min) vs. nontor Max CSI before: {csi_orig[ix_orig]:.4f} at Threshold: {thresholds_orig[ix_orig]:.4f}")
print(f"Pretor (21-30 min) vs. nontor Max CSI after: {csi[ix]:.4f} at Threshold: {thresholds[ix]:.4f}\n")

