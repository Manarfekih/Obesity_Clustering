import streamlit as st
from src.preprocessing import load_and_preprocess, feature_engineer, encode_and_scale
from src.clustering import run_dbscan, run_pca
from pages import overview, cluster_analysis, visualizations, predict_cluster

st.set_page_config(
    page_title="Obesity Clustering App", 
    layout="centered",  # Changed from 'wide' to 'centered' for smaller content
    page_icon="🍎"
)

# Custom CSS to limit max width
st.markdown("""
<style>
.block-container {
    max-width: 800px !important;
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Load custom CSS
with open("assets/styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header - smaller
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <h1 style="font-size: 1.5rem;">🍽️ Obesity Lifestyle Clustering</h1>
    <p style="font-size: 0.8rem; color: #666;">Understanding eating habits & physical activity patterns</p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "📂 Upload CSV", 
    type=["csv"],
    help="Upload your obesity dataset"
)

if uploaded_file:
    with st.spinner("Processing..."):
        df = load_and_preprocess(uploaded_file)
        df = feature_engineer(df)
        X_enc, X_scaled, scaler = encode_and_scale(df)
        labels = run_dbscan(X_scaled)
        df['cluster'] = labels
        X_pca, evr = run_pca(X_scaled)
    
    # Smaller buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Overview", use_container_width=True):
            st.session_state.page = "overview"
    with col2:
        if st.button("🔬 Clusters", use_container_width=True):
            st.session_state.page = "clusters"
    with col3:
        if st.button("📈 Visuals", use_container_width=True):
            st.session_state.page = "visuals"
    with col4:
        if st.button("🔮 Predict", use_container_width=True):
            st.session_state.page = "predict"
    
    if "page" not in st.session_state:
        st.session_state.page = "overview"
    
    st.divider()
    
    # Show selected page
    if st.session_state.page == "overview":
        overview.show_overview(df, X_enc, X_scaled, labels)
    elif st.session_state.page == "clusters":
        cluster_analysis.show_cluster_analysis(df, labels)
    elif st.session_state.page == "visuals":
        visualizations.show_visualizations(df, X_enc, X_scaled, X_pca, evr, labels)
    elif st.session_state.page == "predict":
        predict_cluster.show_predict_cluster(df, X_enc, X_scaled, scaler, labels)

else:
    st.info("📁 Upload a CSV file to begin")