# Unsupervised_Clustering

**Customer Segmentation Using Clustering**

This project segments mall customers into distinct groups based on their spending behavior and income, using unsupervised machine learning (K-Means Clustering). The goal is to help businesses target customers more effectively.

**What It Does**

- Loads customer data (annual income and spending score)

- Applies K-Means clustering to group customers

- Visualizes clusters using a scatter plot

- Provides a Streamlit interface for exploration

**Tools Used**

- Python

- Pandas, NumPy, scikit-learn

- Seaborn, Matplotlib

- Streamlit

**How It Works**

- The dataset is preprocessed

- Optimal number of clusters is determined using the Elbow Method

- K-Means clustering is applied

- Results are visualized and served through a Streamlit app

**Files**

- app.py — Streamlit app

- mall_customers.csv — Dataset

- requirements.txt — Dependencies

**How to Run**

- Clone the repository

- Install dependencies: pip install -r requirements.txt

- Run: streamlit run app.py

**Note**

This project demonstrates basic clustering. It’s ideal for understanding customer segmentation concepts, not production use.

