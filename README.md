# 🧬 Clustering Individuals Based on Eating Habits and Lifestyles

> **Unsupervised Machine Learning — Lifestyle Segmentation Project**  
> UCI Obesity Dataset · KMeans · Hierarchical Clustering · DBSCAN · Streamlit

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Objective](#-business-objective)
3. [Dataset Description](#-dataset-description)
4. [Project Structure](#-project-structure)
5. [Methodology](#-methodology)
6. [Results & Cluster Profiles](#-results--cluster-profiles)
7. [Model Comparison](#-model-comparison)
8. [Key Findings](#-key-findings)
9. [Streamlit Application](#-streamlit-application)
10. [How to Run](#-how-to-run)
11. [Requirements](#-requirements)

---

## 🔍 Project Overview

This project applies **unsupervised machine learning** to discover hidden lifestyle profiles within a population dataset of 2,103 individuals. Rather than predicting obesity labels (a supervised task), the goal is to **identify natural groupings** of people based on their eating habits, physical activity, and lifestyle choices , without using any target labels during training.

The obesity labels (`NObeyesdad`) are used **only after clustering** to validate and interpret the discovered groups.

---

## 🎯 Business Objective

| | |
|---|---|
| **Business Goal** | Understand distinct lifestyle and eating behavior patterns associated with obesity risk |
| **Technical Goal** | Apply clustering algorithms (KMeans, Hierarchical, DBSCAN) to segment individuals |
| **Key Question** | *Can we identify distinct lifestyle profiles from behavioral data alone  without using obesity labels?*
*What eating and activity patterns characterize different population segments?*
|

**Why this matters:**
- Enables targeted health interventions for specific lifestyle groups
- Identifies at-risk populations before clinical obesity diagnosis
- Provides actionable insights for public health campaigns

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| **Source** | [UCI Machine Learning Repository]
| **Instances** | 2,174 (raw) → 2,103 (after cleaning) |
| **Features** | 17 columns (16 features + 1 target) |
| **Type** | Mixed — numerical + categorical |
| **Domain** | Health / Nutrition |

### Feature Groups

| Group | Features |
|---|---|
| **Demographics** | `Gender`, `Age`, `Height`, `Weight` |
| **Health Background** | `family_history_with_overweight` |
| **Eating Habits** | `FAVC`, `FCVC`, `NCP`, `CAEC`, `CH2O` |
| **Lifestyle** | `SMOKE`, `SCC`, `FAF`, `TUE`, `CALC` |
| **Transportation** | `MTRANS` |
| **Target (post-clustering only)** | `NObeyesdad` |

### Feature Details

| Feature | Description | Values |
|---------|-------------|--------|
| **Gender** | Biological sex | Male / Female |
| **Age** | Age in years | Numerical |
| **Height** | Height in meters | Numerical |
| **Weight** | Weight in kg | Numerical |
| **family_history_with_overweight** | Genetic risk factor | Yes / No |
| **FAVC** | Frequent high-calorie food consumption | Yes / No |
| **FCVC** | Vegetable consumption frequency | 1 (low) → 3 (high) |
| **NCP** | Number of main meals per day | Numerical |
| **CAEC** | Snacking between meals | No / Sometimes / Frequently / Always |
| **CH2O** | Daily water intake | 1 (low) → 3 (high) |
| **SMOKE** | Smoking status | Yes / No |
| **SCC** | Calorie consumption monitoring | Yes / No |
| **FAF** | Physical activity frequency | 0 (low) → 3 (high) |
| **TUE** | Daily screen time | 0 (low) → 2 (high) |
| **CALC** | Alcohol consumption | No / Sometimes / Frequently |
| **MTRANS** | Primary transportation mode | Walking / Bike / Public_Transportation / Automobile |

### Engineered Features

| Feature | Formula | Meaning |
|---|---|---|
| `BMI` | `Weight / Height²` | Body Mass Index |
| `Activity_Score` | `FAF + MTRANS_score − TUE`, clipped [0,3] | Combined physical activity indicator |
| `Risk_Score` | `FAVC_bin + CAEC_score`, clipped [0,3] | Combined dietary risk indicator |

---



## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (`01_EDA.ipynb`)

- **Dataset overview**: shape, data types, column inspection
- **Data quality audit**: 66 duplicates detected, missing values in Age (110), Height (131), FAF (154), CAEC (130)
- **Dirty data discovered**: inconsistent capitalization in Gender (`male`, `MALE`, `m`, `M`), mixed formats in FAVC, NCP stored as string
- **Numerical distributions**: histograms and boxplots for Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE — revealed unrealistic values (Height > 3m, negative FAF)
- **Categorical distributions**: countplots for Gender, FAVC, CAEC, MTRANS, CALC
- **Target distribution**: NObeyesdad breakdown — Obese categories are the most frequent
- **Correlation analysis**: heatmap confirmed all numerical features are low-correlated — no redundant features to drop

### Step 2 — Preprocessing (`02_Modeling.ipynb`)

| Step | Action |
|---|---|
| Deduplication | Dropped 66 + 5 additional duplicates (post-imputation) |
| Type coercion | `NCP` was stored as object → converted to float64 |
| Gender normalization | Standardized 8 variants → `Male` / `Female` |
| Range clamping | Height > 3m ÷ 100, Age < 10 or > 110 → NaN, FAF < 0 → NaN |
| Missing values — numerical | Filled with **median** (robust to outliers) |
| Missing values — categorical | Filled with **mode** |
| Final shape | **2,103 rows × 17 columns** |

### Step 3 — Feature Engineering

```python
# BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Activity Score: combines exercise frequency, transport mode, screen time
MTRANS_score = {'Walking': 3, 'Bike': 2, 'Public_Transportation': 1, 'Automobile': 0, 'Motorbike': 0}
df['Activity_Score'] = (df['FAF'] + df['MTRANS_score'] - df['TUE']).clip(0, 3)

# Risk Score: combines high-calorie food intake and snacking behavior
df['Risk_Score'] = (FAVC_binary + CAEC_ordinal).clip(0, 3)
```

### Step 4 — Encoding & Scaling

```python
X = df.drop(columns=['NObeyesdad'])           # Drop target — not used in training
X_encoded = pd.get_dummies(X, drop_first=True) # One-hot encoding → 26 features
X_scaled = StandardScaler().fit_transform(X_encoded)  # Critical for distance-based algorithms
```

### Step 5 — Dimensionality Analysis (PCA)

| Threshold | Components needed |
|---|---|
| 85% variance | 15 components |
| 95% variance | 19 components |
| 2D visualization | 2 components (23.1% variance) |

> 23% variance in 2D is expected for high-dimensional mixed data — the low 2D coverage does not invalidate the clusters, it only means the 2D projection is a simplified view.

### Step 6 — Modeling

#### Model 1 — K-Means
- **k selection**: Elbow method + Silhouette score over k ∈ [2, 10]
- **Best k = 2** (silhouette = 0.2293, inertia elbow between k=2 and k=3)
- **n_init stability test**: score stabilizes at n_init=10; n_init=15 used for final model

#### Model 2 — Hierarchical Clustering (Agglomerative)
- **Three linkages tested**: Ward (0.1301), Complete (0.6507*), Average (0.6507*)
- **Ward chosen**: Complete linkage achieved inflated silhouette by pushing 2,096/2,103 points into one cluster (degenerate result /  not meaningful)
- **Dendrogram**: 200-sample subset used for visualization; threshold = 0.7 × max distance

#### Model 3 — DBSCAN
- **eps selection**: k-distance graph (elbow method) with k ∈ {5, 10, 20, 52}
- **Grid search**: eps ∈ [5.28, 6.23] × min_samples ∈ {19, 20, 21}
- **Best params**: eps = 6.23, min_samples = 19 (silhouette = 0.3275 in grid → 0.3412 on valid points)
- **Noise handling**: 48 points (2.3%) flagged as outliers — silhouette computed on non-noise points only

---

## 📈 Results & Cluster Profiles

### DBSCAN — Best Model (4 clusters)

| Cluster | Size | Profile | Risk Score | Activity Score | Dominant Obesity Level |
|---|---|---|---|---|---|
| **0** | 1,934 (93.2%) | 🔴 High-Risk Group | 2.02 / 3 | 0.84 / 3 | Obesity Type I/II/III |
| **1** | 30 (1.4%) | 🟡 Older High-Risk Drinkers | 1.87 / 3 | 0.97 / 3 | Obesity Type II (46.7%) |
| **2** | 46 (2.2%) | 🟢 Healthiest Group | 0.87 / 3 | 1.35 / 3 | Overweight Level I (54.3%) |
| **3** | 45 (2.2%) | 🏃 Young & Very Active | 1.91 / 3 | 2.69 / 3 | Normal Weight (60%) |

**Cluster interpretations:**
- **Cluster 0**: The dominant group : high FAVC (89%), frequent snacking, moderate activity, family history of overweight. Strongly linked to obesity.
- **Cluster 1**: Older individuals (avg 28y), 100% consume alcohol "sometimes", no abstainers. Highest share of Obesity Type II.
- **Cluster 2**: Lowest risk score, no frequent snacking, highest activity — the healthiest behavioral profile despite mild overweight.
- **Cluster 3**: Youngest group (avg 20.8y), very active (Activity Score 2.69), lowest high-calorie food intake (60%). Predominantly Normal Weight.

---

## 🏆 Model Comparison

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Clusters | Notes |
|---|---|---|---|---|
| **DBSCAN** ✅ | **0.3412** | **1.1850** | 4 | Best on both metrics. Noise-tolerant. |
| K-Means | 0.2293 | 2.6671 | 2 | Only 2 clusters — limited interpretability |
| Hierarchical (Ward) | 0.1301 | 3.1020 | 2 | Weakest separation |

**Why DBSCAN wins:**
1. Does not require specifying k upfront
2. Naturally handles outliers (2.3% noise)
3. Finds arbitrarily-shaped clusters
4. Best quantitative scores on both Silhouette and Davies-Bouldin

### Top Features (PCA Loadings)

1. `CALC_Sometimes` — alcohol consumption pattern
2. `Risk_Score` — engineered dietary risk
3. `CALC_No` — abstaining from alcohol
4. `Age` — demographic factor
5. `MTRANS_Public_Transportation` — sedentary transport indicator
6. `Height` — body measurement
7. `CAEC_Frequently` — frequent between-meal snacking
8. `Activity_Score` — engineered physical activity composite

---



## 🌐 Streamlit Application

An interactive web application allows users to:
- Upload the dataset and explore it interactively
- View cluster profiles, heatmaps, radar charts, and PCA scatter plots
- **Predict which cluster their own profile belongs to** using k-NN majority vote


```bash
streamlit run app.py
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/obesity-clustering.git
cd obesity-clustering
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks
```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Modeling.ipynb
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

---

## 👤 Author

**[Manar El Fakih Romdhane]** — ML Project · 2025/2026  
Supervised by: [Khemais Abdallah]  
Dataset: [UCI Obesity Dataset]

---

*This project was developed as part of a Machine Learning course. The obesity labels were used exclusively for post-clustering interpretation , not as training targets.*
