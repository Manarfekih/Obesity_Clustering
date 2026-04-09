import streamlit as st
import matplotlib.pyplot as plt

def show_overview(df_clean, X_enc, X_scaled, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    # Metrics in compact row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", f"{len(df_clean):,}")
    m2.metric("Features", str(X_enc.shape[1]))
    m3.metric("Clusters", str(n_clusters))
    m4.metric("Noise", str(n_noise))
    
    # BMI Distribution - TINY chart
    st.markdown("### BMI Distribution")
    bmi_counts = df_clean['BMI_Category'].value_counts()
    
    fig, ax = plt.subplots(figsize=(4, 2.5))  # Very small
    bars = ax.bar(bmi_counts.index, bmi_counts.values, 
                  color=['#FF6B6B', '#FFB86B', '#6BCB77', '#4D96FF'],
                  edgecolor='white', linewidth=1)
    ax.set_facecolor('#FAFAFA')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)