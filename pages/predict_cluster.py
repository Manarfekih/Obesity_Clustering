import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.constants import CLUSTER_META

def show_predict_cluster(df_clean, X_enc, X_scaled, scaler, labels):
    st.markdown("### Enter your details")
    
    with st.form("prediction_form"):
        # 2 columns for compact form
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 14, 70, 25)
            height = st.number_input("Height (m)", 1.45, 2.00, 1.70, 0.01)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
            family_history = st.selectbox("Family overweight history", ["No", "Yes"])
            favc = st.selectbox("Frequent high-calorie food", ["No", "Yes"])
        
        with col2:
            fcvc = st.slider("Vegetable consumption (1-3)", 1, 3, 2)
            ncp = st.slider("Main meals/day", 1, 4, 3)
            caec = st.selectbox("Snacking", ["No", "Sometimes", "Frequently", "Always"])
            ch2o = st.slider("Water intake (1-3)", 1, 3, 2)
            faf = st.slider("Physical activity (0-3)", 0, 3, 1)
            tue = st.slider("Screen time (0-2)", 0, 2, 1)
            mtrans = st.selectbox("Main transport", ["Public_Transportation", "Automobile", "Walking", "Bike"])
        
        submitted = st.form_submit_button("🔍 Predict My Cluster", use_container_width=True)
    
    if submitted:
        with st.spinner("Analyzing..."):
            # Build user data (same logic but simplified)
            user_dict = {
                'Age': age, 'Height': height, 'Weight': weight,
                'FCVC': fcvc, 'NCP': ncp, 'CH2O': ch2o, 'FAF': faf, 'TUE': tue,
                'Gender': gender, 'family_history_with_overweight': family_history,
                'FAVC': favc, 'CAEC': caec, 'SMOKE': "No", 'SCC': "No",
                'CALC': "No", 'MTRANS': mtrans
            }
            
            user_df = pd.DataFrame([user_dict])
            user_df['BMI'] = user_df['Weight'] / (user_df['Height'] ** 2)
            
            # Quick encoding
            mtrans_map = {'Walking': 3, 'Bike': 2, 'Public_Transportation': 1, 'Automobile': 0}
            user_df['MTRANS_score'] = user_df['MTRANS'].map(mtrans_map).fillna(0)
            user_df['Activity_Score'] = (user_df['FAF'] + user_df['MTRANS_score'] - user_df['TUE']).clip(0, 3)
            
            user_df['FAVC_bin'] = (user_df['FAVC'] == "Yes").astype(int)
            caec_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
            user_df['CAEC_score'] = user_df['CAEC'].map(caec_map).fillna(1)
            user_df['Risk_Score'] = (user_df['FAVC_bin'] + user_df['CAEC_score']).clip(upper=3)
            
            # Encode
            user_enc = pd.get_dummies(user_df.drop(columns=['MTRANS_score', 'FAVC_bin', 'CAEC_score'], errors='ignore'))
            for col in X_enc.columns:
                if col not in user_enc.columns:
                    user_enc[col] = 0
            user_enc = user_enc[X_enc.columns]
            
            # Predict
            user_scaled = scaler.transform(user_enc)
            nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
            dists, idxs = nbrs.kneighbors(user_scaled)
            neighbour_labels = labels[idxs[0]]
            valid_neighbours = neighbour_labels[neighbour_labels != -1]
            pred_cluster = -1 if len(valid_neighbours) == 0 else pd.Series(valid_neighbours).mode()[0]
            
            # Show result
            meta = CLUSTER_META.get(pred_cluster, CLUSTER_META[-1])
            st.success(f"✅ You belong to **Cluster {pred_cluster} — {meta['label']}**")
            st.info(f"💡 {meta['desc']}")