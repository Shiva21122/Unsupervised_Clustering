import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Load models and data ───────────────────────────────────────────
@st.cache_resource
def load_models_and_data():
    here = os.getcwd()
    with open(os.path.join(here, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(here, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    df = pd.read_csv(os.path.join(here, "mall_customers.csv"))
    return scaler, kmeans, df

scaler, kmeans, df = load_models_and_data()

# ─── Prepare full‐data clusters ──────────────────────────────────────
df["Gender_Num"] = df["Gender"].map({"Male": 1, "Female": 0})
X_full = df[["Gender_Num", "Age", "Annual_Income", "Spending_Score"]].values
X_full_scaled = scaler.transform(X_full)
df["Cluster"] = kmeans.predict(X_full_scaled)

# ─── App title and scatter plot ─────────────────────────────────────
st.title("Unsupervised Clustering Demo")
st.header("Cluster Visualization: Income vs Spending Score")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    cmap="tab10",
    s=50,
    edgecolor="k",
    alpha=0.7
)
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments by Income & Spending Score")
handles, _ = scatter.legend_elements()
labels = [f"Cluster {i}" for i in range(kmeans.n_clusters)]
ax.legend(handles, labels, title="Clusters", loc="best")
st.pyplot(fig)

st.markdown("---")

# ─── Interactive prediction ─────────────────────────────────────────
st.header("Predict Cluster for a New Customer")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1, format="%d")
income = st.number_input("Annual Income", min_value=0, value=50000, step=1, format="%d")
score = st.number_input(
    "Spending Score",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    format="%d"
)

if st.button("Assign Cluster"):
    gender_num = 1 if gender == "Male" else 0
    features = np.array([[gender_num, age, income, score]])
    X_new_scaled = scaler.transform(features)
    cluster_new = kmeans.predict(X_new_scaled)[0]
    st.success(f"🗂️ This customer belongs to **cluster {cluster_new}**")
    
    dists = kmeans.transform(X_new_scaled)[0]
    dist_series = pd.Series(dists, index=[f"Cluster {i}" for i in range(len(dists))])
    st.bar_chart(dist_series)
