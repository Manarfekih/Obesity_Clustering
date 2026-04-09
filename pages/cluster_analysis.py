import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.constants import CLUSTER_META

def show_cluster_analysis(df_clean, labels):
    cluster_ids = sorted([c for c in set(labels) if c != -1])
    
    # Compact cluster cards
    for i in range(0, len(cluster_ids), 2):
        cols = st.columns(2)
        for j, cid in enumerate(cluster_ids[i:i+2]):
            meta = CLUSTER_META.get(cid, CLUSTER_META[-1])
            size = (labels == cid).sum()
            pct = size / len(labels) * 100
            
            with cols[j]:
                st.markdown(f"""
                <div style="background:{meta['color']}15; border-left: 2px solid {meta['border']}; 
                            padding: 0.4rem; border-radius: 4px; margin-bottom: 0.4rem;">
                    <b style="font-size:0.8rem">{meta['emoji']} Cluster {cid} — {meta['label']}</b><br>
                    <span style="font-size:0.7rem">{size:,} people ({pct:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Feature heatmap - TINY
    st.markdown("### Feature Profiles")
    profile_cols = ['Risk_Score', 'Activity_Score', 'FAF', 'BMI', 'Age', 'CH2O', 'FCVC', 'TUE']
    valid_df = df_clean[df_clean['cluster'] != -1].copy()
    profile = valid_df.groupby('cluster')[profile_cols].mean()
    
    fig, ax = plt.subplots(figsize=(5, 2))  # Very small
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    sns.heatmap(profile_norm.T, annot=profile.T, fmt='.1f', cmap='RdYlGn',
                ax=ax, linewidths=0.3, cbar=False,  # Remove colorbar to save space
                annot_kws={'size': 5})
    ax.tick_params(labelsize=5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    
    # Obesity heatmap - TINY
    st.markdown("### Obesity Levels")
    valid_df['cluster_str'] = valid_df['cluster'].astype(str)
    crosstab = pd.crosstab(valid_df['cluster_str'], valid_df['NObeyesdad'], normalize='index') * 100
    
    fig2, ax2 = plt.subplots(figsize=(5, 2))  # Very small
    sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                cbar=False, annot_kws={'size': 4})
    ax2.tick_params(labelsize=5)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=False)