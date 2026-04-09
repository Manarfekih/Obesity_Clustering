import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import streamlit as st

@st.cache_data
def run_dbscan(X_scaled, eps=6.23, min_samples=19):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels

@st.cache_data
def run_pca(X_scaled, n=2):
    pca = PCA(n_components=n)
    return pca.fit_transform(X_scaled), pca.explained_variance_ratio_