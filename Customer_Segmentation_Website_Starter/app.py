import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation Using Clustering")
st.caption("PDS PBL • Unsupervised Learning • K-Means")

st.write(
    "Upload customer data or use the sample dataset. Choose numeric features, "
    "select the number of clusters, and run K-Means to group similar customers."
)

@st.cache_data
def load_sample():
    return pd.read_csv("sample_customers.csv")

with st.sidebar:
    st.header("1. Choose data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    st.header("2. Clustering settings")
    st.caption("Choose numeric columns that describe customer behavior.")

try:
    df = pd.read_csv(uploaded) if uploaded is not None else load_sample()
except Exception as e:
    st.error(f"Could not load the data: {e}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

st.subheader("Customer Dataset")
st.dataframe(df.head(100), use_container_width=True)
st.caption(f"Rows: {len(df)} • Columns: {len(df.columns)}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Your CSV needs at least two numeric columns for this demo. Example: Annual Income and Spending Score.")
    st.stop()

with st.sidebar:
    default_features = numeric_cols[:min(2, len(numeric_cols))]
    selected_features = st.multiselect(
        "Numeric features",
        options=numeric_cols,
        default=default_features
    )
    k = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=3)
    run = st.button("Run Customer Segmentation", type="primary", use_container_width=True)

if not run:
    st.info("Select at least two numeric features in the sidebar, then click **Run Customer Segmentation**.")
    st.stop()

if len(selected_features) < 2:
    st.warning("Please select at least two numeric features.")
    st.stop()

work = df[selected_features].copy()
for col in selected_features:
    work[col] = pd.to_numeric(work[col], errors="coerce")

valid_mask = work.notna().all(axis=1)
work_valid = work.loc[valid_mask]
if len(work_valid) < k:
    st.error(f"At least {k} complete customer rows are needed. Only {len(work_valid)} are available.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(work_valid)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

results = df.loc[valid_mask].copy()
results["Cluster"] = labels.astype(int)
sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 and len(work_valid) > k else None

st.subheader("Segmentation Results")
c1, c2, c3 = st.columns(3)
c1.metric("Customers analyzed", f"{len(results)}")
c2.metric("Clusters created", f"{k}")
c3.metric("Silhouette score", f"{sil:.3f}" if sil is not None else "Not available")

st.write("Each point below represents a customer. Colors indicate the cluster assigned by K-Means.")
x_feature, y_feature = selected_features[0], selected_features[1]
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(results[x_feature], results[y_feature], c=results["Cluster"], cmap="viridis", s=55, alpha=0.8)
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title("Customer Clusters")
ax.grid(True, alpha=0.25)
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Cluster Summary")
summary = results.groupby("Cluster")[selected_features].mean().round(2)
summary.insert(0, "Number of Customers", results.groupby("Cluster").size())
st.dataframe(summary, use_container_width=True)

st.subheader("Customers with Cluster Labels")
st.dataframe(results, use_container_width=True)

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    data=csv_bytes,
    file_name="customer_segmentation_results.csv",
    mime="text/csv"
)

st.subheader("How to interpret the clusters")
st.write(
    "Compare the average values in the Cluster Summary. For example, if one cluster has "
    "higher spending scores or annual income than the others, it may represent a higher-spending "
    "customer group. The labels are data-dependent; K-Means does not automatically know which "
    "group is 'high-value' or 'at-risk'."
)

with st.expander("What is K-Means clustering?"):
    st.write(
        "K-Means is an unsupervised machine-learning algorithm. It groups records by similarity "
        "in selected numeric features. You choose K (the number of groups), and the algorithm "
        "assigns each customer to one cluster."
    )

st.caption("Educational demo. Use anonymized customer data and avoid uploading personal identifiers.")
