import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def show_visualizations(df_clean, X_enc, X_scaled, X_pca, evr, labels):
    # PCA Scatter - TINY
    with st.expander("🎯 PCA Projection"):
        st.caption(f"PC1: {evr[0]*100:.1f}% | PC2: {evr[1]*100:.1f}%")
        
        cluster_colors = {0: '#FF6B6B', 1: '#FFB86B', 2: '#6BCB77', 3: '#4D96FF', -1: '#CCCCCC'}
        
        fig, ax = plt.subplots(figsize=(5, 3.5))  # Small
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            alpha = 0.3 if lbl == -1 else 0.7
            label = 'Noise' if lbl == -1 else f'C{lbl}'
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=cluster_colors.get(lbl, '#999'), alpha=alpha, s=8, label=label)
        
        ax.set_facecolor('#FAFAFA')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=6, loc='best', ncol=2)
        ax.tick_params(labelsize=6)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
    
    # Radar Chart - TINY
    with st.expander("📊 Radar Chart"):
        features = ['Risk_Score', 'Activity_Score', 'FAF', 'FCVC', 'CH2O', 'TUE']
        valid_df = df_clean[df_clean['cluster'] != -1]
        cp = valid_df.groupby('cluster')[features].mean()
        cp_norm = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)
        
        N = len(features)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        fig2 = plt.figure(figsize=(3.5, 3.5))  # Small square
        ax2 = fig2.add_subplot(111, polar=True)
        colors = ['#FF6B6B', '#FFB86B', '#6BCB77', '#4D96FF']
        
        for i, cid in enumerate(cp_norm.index):
            vals = cp_norm.loc[cid].tolist() + [cp_norm.loc[cid].tolist()[0]]
            ax2.plot(angles, vals, color=colors[i % len(colors)], linewidth=1, label=f'C{cid}')
            ax2.fill(angles, vals, color=colors[i % len(colors)], alpha=0.1)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(features, fontsize=5)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=5)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=False)
    
    # Feature Importance - TINY
    with st.expander("⭐ Feature Importance"):
        pca5 = PCA(n_components=5)
        pca5.fit(X_scaled)
        loadings = pd.DataFrame(np.abs(pca5.components_).T, index=X_enc.columns)
        loadings_sum = loadings.sum(axis=1).sort_values(ascending=False).head(8)
        
        fig3, ax3 = plt.subplots(figsize=(4, 2.5))  # Small
        ax3.barh(loadings_sum.index[::-1], loadings_sum.values[::-1], 
                color='#4D96FF', edgecolor='white', height=0.6)
        ax3.set_facecolor('#FAFAFA')
        ax3.spines[['top', 'right', 'bottom']].set_visible(False)
        ax3.tick_params(axis='both', labelsize=6)
        ax3.set_xlabel("Loading", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=False)