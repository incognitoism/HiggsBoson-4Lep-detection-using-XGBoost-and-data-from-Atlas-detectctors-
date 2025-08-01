# ==============================================================================
# AI for Particle Physics - Part 1: Data Processing and Model Training
#
# This script performs all the necessary steps to process the raw data,
# train the XGBoost and DNN models, and save all the resulting assets
# (DataFrames, models, scaler) to disk for use in the Streamlit app.
#
# This script only needs to be run once.
# ==============================================================================

import os
# --- FIX: Disable GPU to avoid TensorFlow Metal plugin errors on some systems ---
# This line MUST be at the top, before TensorFlow is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import uproot
import pandas as pd
import numpy as np
import glob
import awkward as ak
import vector
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Starting Data Processing and Model Training ---")

# --- 1. Data Processing Function ---
def process_file(file_path, is_mc=False):
    """Reads a single ROOT file and returns a clean, flat pandas DataFrame."""
    branches_to_read = ["lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_charge", "lep_type"]
    if is_mc:
        branches_to_read.extend(["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON"])

    try:
        with uproot.open(file_path) as file:
            tree = file["mini"]
            events = tree.arrays(branches_to_read, library="ak")
    except Exception as e:
        print(f"Warning: Could not process {os.path.basename(file_path)}. Reason: {e}")
        return None

    four_lepton_mask = ak.num(events.lep_pt) == 4
    events = events[four_lepton_mask]
    charge_mask = ak.sum(events.lep_charge, axis=1) == 0
    events = events[charge_mask]

    if len(events) == 0: return None

    leptons = vector.zip({"pt": events.lep_pt, "eta": events.lep_eta, "phi": events.lep_phi, "E": events.lep_E})
    four_lepton_system = leptons[:, 0] + leptons[:, 1] + leptons[:, 2] + leptons[:, 3]

    df = pd.DataFrame()
    for i in range(4):
        # Convert to GeV during creation
        df[f'lep_pt_{i}'] = leptons.pt[:, i] / 1000
        df[f'lep_eta_{i}'] = leptons.eta[:, i]
        df[f'lep_phi_{i}'] = leptons.phi[:, i]
        df[f'lep_E_{i}'] = leptons.E[:, i] / 1000
        df[f'lep_charge_{i}'] = events.lep_charge[:, i]
        df[f'lep_type_{i}'] = events.lep_type[:, i]

    df['M_4l'] = four_lepton_system.mass / 1000
    df['pT_4l'] = four_lepton_system.pt / 1000

    if is_mc:
        df['mcWeight'] = events.mcWeight
        df['scaleFactor_PILEUP'] = events.scaleFactor_PILEUP
        df['scaleFactor_ELE'] = events.scaleFactor_ELE
        df['scaleFactor_MUON'] = events.scaleFactor_MUON

    return df

# --- 2. Load, Process, and Label Data ---
print("\nStep 1: Loading and processing data...")
base_dir = "dataset/4lep 2"
mc_dir = os.path.join(base_dir, "MC")
data_dir = os.path.join(base_dir, "Data")

signal_ids = ['ggH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep']
background_ids = ['Zee', 'Zmumu']

# Process Signal
signal_files = [f for f in glob.glob(os.path.join(mc_dir, "*.root")) if any(id_str in f for id_str in signal_ids)]
signal_dfs = [process_file(f, is_mc=True) for f in signal_files]
signal_df = pd.concat([df for df in signal_dfs if df is not None], ignore_index=True)
signal_df['is_signal'] = 1

# Process Background
background_files = [f for f in glob.glob(os.path.join(mc_dir, "*.root")) if any(id_str in f for id_str in background_ids)]
background_dfs = [process_file(f, is_mc=True) for f in background_files]
background_df = pd.concat([df for df in background_dfs if df is not None], ignore_index=True)
background_df['is_signal'] = 0

# Combine into the final mc_df
mc_df = pd.concat([signal_df, background_df], ignore_index=True)
mc_df['total_weight'] = (mc_df['mcWeight'] * mc_df['scaleFactor_PILEUP'] * mc_df['scaleFactor_ELE'] * mc_df['scaleFactor_MUON']).fillna(0)

# Process Real Data
data_files = glob.glob(os.path.join(data_dir, "*.root"))
data_dfs = [process_file(f, is_mc=False) for f in data_files]
data_df = pd.concat([df for df in data_dfs if df is not None], ignore_index=True)

print("Data loading complete.")
print(f"Loaded {len(mc_df)} MC events and {len(data_df)} real data events.")

# --- 3. Prepare Data for Models ---
features = [f'lep_pt_{i}' for i in range(4)] + [f'lep_eta_{i}' for i in range(4)] + \
           [f'lep_phi_{i}' for i in range(4)] + [f'lep_E_{i}' for i in range(4)] + ['M_4l']

X = mc_df[features].astype('float32')
y = mc_df['is_signal'].astype('int32')
weights = mc_df['total_weight'].astype('float32')
weights[weights < 0] = 0 # Handle negative weights

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42, stratify=y)

# --- 4. Train and Save XGBoost Model ---
print("\nStep 2: Training XGBoost model...")
dtrain = xgb.DMatrix(X_train.values, label=y_train.values, weight=w_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values, weight=w_test.values)
params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.1, 'max_depth': 4, 'seed': 42}
bst = xgb.train(params, dtrain, 250, evals=[(dtest, 'test')], early_stopping_rounds=20, verbose_eval=False)
joblib.dump(bst, 'xgb_model.joblib')
print("XGBoost model saved to xgb_model.joblib")

# --- 5. Train and Save DNN Model ---
print("\nStep 3: Training DNN model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def build_dnn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(64), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model_dnn = build_dnn_model(X_train_scaled.shape[1:])
model_dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])
callbacks = [keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)]
model_dnn.fit(X_train_scaled, y_train, sample_weight=w_train.values, validation_split=0.2, epochs=100, batch_size=1024, callbacks=callbacks, verbose=0)
model_dnn.save('dnn_model.keras')
joblib.dump(scaler, 'scaler.joblib')
print("DNN model saved to dnn_model.keras")
print("Scaler saved to scaler.joblib")

# --- 6. Save DataFrames ---
print("\nStep 4: Saving DataFrames...")
# CORRECTED: Save as .parquet instead of .pkl for better compatibility
mc_df.to_parquet("mc_df.parquet")
data_df.to_parquet("data_df.parquet")
print("DataFrames saved to mc_df.parquet and data_df.parquet")

print("\n--- All assets have been created successfully! ---")
