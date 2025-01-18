import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title("Customer Segmentation using K-Means Clustering")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    customer_data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(customer_data.head())

    # Preprocess data
    scaler = StandardScaler()
    X = customer_data.iloc[:, 2:5].values
    X_scaled = scaler.fit_transform(X)

    # Select number of clusters
    num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=6)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=50, random_state=125)
    customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    customer_data['PC1'] = principal_components[:, 0]
    customer_data['PC2'] = principal_components[:, 1]

    # Display cluster data
    st.subheader("Clustered Data Preview")
    st.write(customer_data.head())

    # Cluster visualization
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=customer_data['PC1'],
        y=customer_data['PC2'],
        hue=customer_data['Cluster'],
        palette='viridis',
        s=100,
        ax=ax
    )
    plt.title("Customer Segmentation Clusters (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(fig)

    # Download clustered data
    csv = customer_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv,
        file_name='clustered_customers.csv',
        mime='text/csv'
    )
else:
    st.write("Upload a dataset to proceed!")
