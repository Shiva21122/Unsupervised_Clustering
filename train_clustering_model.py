import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_PATH   = "mall_customers.csv"        # replace with your actual CSV
SCALER_PATH = "scaler.pkl"
MODEL_PATH  = "kmeans_model.pkl"
N_CLUSTERS  = 5                      # tweak or set via elbow/silhouette

# ─── LOAD & PREPARE ───────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found in {os.getcwd()}")

df = pd.read_csv(DATA_PATH)

# Select only the numeric features you clustered on in your notebook.
# e.g. features = ['age', 'income', 'sqft', ...]
features = [col for col in df.columns if df[col].dtype in (np.int64, np.float64)]
X = df[features].copy()

# Handle missing values if needed
X.fillna(X.mean(), inplace=True)

# ─── SCALE ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── TRAIN ────────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans.fit(X_scaled)

# Optional: print a silhouette score to gauge clustering quality
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette score for k={N_CLUSTERS}: {score:.3f}")

# ─── SAVE ARTIFACTS ───────────────────────────────────────────────────────
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"✅ Saved scaler to {SCALER_PATH}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(kmeans, f)
print(f"✅ Saved KMeans model to {MODEL_PATH}")
