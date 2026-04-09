import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file).drop_duplicates()

    # Fix Gender
    df['Gender'] = df['Gender'].str.strip().str.lower().replace({'female':'Female','male':'Male','f':'Female','m':'Male'})

    # Categorical columns
    catego=['CAEC', 'CALC', 'FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']
    for col in catego:
        df[col] = df[col].str.strip().str.capitalize()
    df['MTRANS'] = df['MTRANS'].str.strip()

    # Numeric columns
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix inconsistencies
    df.loc[df['Height'] > 3, 'Height'] /= 100
    df.loc[(df['Age'] < 10) | (df['Age'] > 110), 'Age'] = np.nan
    df.loc[(df['Weight'] < 20) | (df['Weight'] > 200), 'Weight'] = np.nan
    df.loc[(df['FAF'] < 0) | (df['FAF'] > 3), 'FAF'] = np.nan

    # Fill missing values
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def feature_engineer(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / (df['Height']**2)
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0,18.5,25,30,float('inf')],
        labels=['Underweight','Normal','Overweight','Obese'],
        right=False
    )

    mtrans_activity = {'Walking':3,'Bike':2,'Public_Transportation':1,'Automobile':0,'Motorbike':0}
    df['MTRANS_score'] = df['MTRANS'].map(mtrans_activity).fillna(0)
    df['Activity_Score'] = (df['FAF'] + df['MTRANS_score'] - df['TUE']).clip(0,3).astype(int)
    df['FAVC_bin'] = (df['FAVC'].str.lower() == 'yes').astype(int)
    df['CAEC_score'] = df['CAEC'].map({'No':0,'Sometimes':1,'Frequently':2,'Always':3}).fillna(1)
    df['Risk_Score'] = (df['FAVC_bin'] + df['CAEC_score']).clip(upper=3)
    return df

def encode_and_scale(df):
    X = df.drop(columns=['NObeyesdad','BMI_Category','MTRANS_score','FAVC_bin','CAEC_score'], errors='ignore')
    X_enc = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    return X_enc, X_scaled, scaler