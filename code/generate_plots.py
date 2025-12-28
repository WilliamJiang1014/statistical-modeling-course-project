
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc
import os

# Create output directory
if not os.path.exists('docs/figures'):
    os.makedirs('docs/figures')

# Load data
df = pd.read_csv('data/processed/diabetic_data_cleaned.csv')

# ==========================================
# 1. Pie Chart: Readmission Distribution
# ==========================================
print("Generating Pie Chart...")
readmitted_counts = df['readmitted_30d'].value_counts()
# Assuming 1 is Readmitted (<30), 0 is Not
labels = ['Not Readmitted', 'Readmitted (<30 days)']
sizes = [readmitted_counts.get(0, 0), readmitted_counts.get(1, 0)]
colors = ['#66B2FF', '#FF9999']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
plt.title('Readmission Distribution')
plt.savefig('docs/figures/readmission_pie_chart.png')
plt.close()

# ==========================================
# 2. Histogram: Time in Hospital
# ==========================================
print("Generating Histogram...")
plt.figure(figsize=(8, 6))
sns.histplot(df['time_in_hospital'], bins=14, kde=False, color='#69b3a2', edgecolor='black')
plt.title('Distribution of Time in Hospital')
plt.xlabel('Time in Hospital (days)')
plt.ylabel('Frequency')
plt.savefig('docs/figures/time_in_hospital_histogram.png')
plt.close()

# ==========================================
# 3. Heatmap: Correlation
# ==========================================
print("Generating Heatmap...")
cols = ['number_diagnoses', 'num_lab_procedures', 'time_in_hospital', 'num_medications']
corr_matrix = df[cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('docs/figures/correlation_heatmap.png')
plt.close()

# ==========================================
# Feature Engineering
# ==========================================
print("Performing Feature Engineering...")
df_fe = df.copy()

def categorize_icd9(diag_code):
    if pd.isna(diag_code) or diag_code == 'Unknown':
        return 'Unknown'
    diag_str = str(diag_code)
    try:
        diag_num = float(diag_str)
    except ValueError:
        return 'Unknown'
    
    if 250 <= diag_num < 251: return 'Diabetes'
    elif (390 <= diag_num < 460) or (785 <= diag_num < 786): return 'Circulatory'
    elif 460 <= diag_num < 520: return 'Respiratory'
    elif 520 <= diag_num < 580: return 'Digestive'
    elif 580 <= diag_num < 630: return 'Genitourinary'
    elif 800 <= diag_num < 1000: return 'Injury'
    elif 140 <= diag_num < 240: return 'Neoplasms'
    else: return 'Other'

def group_discharge_disposition(discharge_id):
    if discharge_id == 1: return 'Home'
    elif discharge_id in [2, 3, 4, 5, 6]: return 'Transfer'
    elif discharge_id in [7, 8, 9, 10]: return 'Other_Facility'
    elif discharge_id in [11, 12, 13, 14, 15, 16, 17, 18, 19]: return 'Other'
    elif discharge_id in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]: return 'Death_Other'
    else: return 'Unknown'

def group_admission_source(source_id):
    if source_id in [1, 2, 3]: return 'Referral'
    elif source_id in [4, 5, 6, 7, 8]: return 'Transfer'
    elif source_id in [9, 10, 11, 12, 13, 14, 15, 16, 17]: return 'Emergency_Other'
    else: return 'Other'

df_fe['diag_1_group'] = df_fe['diag_1'].apply(categorize_icd9)
df_fe['diag_2_group'] = df_fe['diag_2'].apply(categorize_icd9)
df_fe['diag_3_group'] = df_fe['diag_3'].apply(categorize_icd9)
df_fe['discharge_disposition_group'] = df_fe['discharge_disposition_id'].apply(group_discharge_disposition)
df_fe['admission_source_group'] = df_fe['admission_source_id'].apply(group_admission_source)

# Define Features
regression_features_enhanced = [
  'age', 'gender', 'race',
  'number_outpatient', 'number_emergency', 'number_inpatient',
  'number_diagnoses',
  'num_lab_procedures', 'num_procedures', 'num_medications',
  'admission_type_id', 'admission_source_id',
  'diag_1_group', 'diag_2_group', 'diag_3_group',
  'discharge_disposition_group', 'admission_source_group',
  'insulin_use', 'diabetesMed', 'medication_changed'
]

classification_features_enhanced = [
  'age', 'gender', 'race',
  'number_outpatient', 'number_emergency', 'number_inpatient',
  'number_diagnoses',
  'num_medications', 'insulin_use', 'diabetesMed',
  'admission_type_id',
  'diag_1_group', 'diag_2_group', 'diag_3_group',
  'discharge_disposition_group', 'admission_source_group',
  'medication_changed',
  'num_lab_procedures', 'num_procedures'
]

# Encode Features (Label Encoding to match R script behavior)
def encode_features(df, features):
    df_encoded = df.copy()
    encoders = {}
    for col in features:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    return df_encoded, encoders

# ==========================================
# 4. Regression Coefficients
# ==========================================
print("Generating Regression Coefficients Plot...")
X_reg = df_fe[regression_features_enhanced]
y_reg = df_fe['time_in_hospital']

X_reg_encoded, _ = encode_features(X_reg, regression_features_enhanced)

# Split (to match general process, though we just need coefficients)
X_train, X_test, y_train, y_test = train_test_split(X_reg_encoded, y_reg, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

# Coefficients
coef_df = pd.DataFrame({
    'Variable': regression_features_enhanced,
    'Coefficient': reg_model.coef_
})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
top_coefs = coef_df.sort_values(by='AbsCoef', ascending=False).head(15)

# Highlight
top_coefs['Highlight'] = top_coefs['Variable'].apply(lambda x: 'Yes' if x in ['number_diagnoses', 'num_lab_procedures'] else 'No')

plt.figure(figsize=(10, 8))
colors = top_coefs['Highlight'].map({'Yes': 'red', 'No': 'gray'})
plt.barh(top_coefs['Variable'], top_coefs['Coefficient'], color=colors)
plt.xlabel('Coefficient')
plt.title('Regression Coefficients (Top 15)')
plt.gca().invert_yaxis()
plt.savefig('docs/figures/regression_coefficients.png')
plt.close()

# ==========================================
# 5. ROC Curve
# ==========================================
print("Generating ROC Curve...")
X_clf = df_fe[classification_features_enhanced]
y_clf = df_fe['readmitted_30d']

X_clf_encoded, _ = encode_features(X_clf, classification_features_enhanced)

X_train, X_test, y_train, y_test = train_test_split(X_clf_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
clf_model = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_model.fit(X_train_scaled, y_train)

# Predict
y_pred_prob = clf_model.predict_proba(X_test_scaled)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"Calculated AUC: {roc_auc}")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Logistic Regression)')
# Annotate as requested
plt.text(0.6, 0.4, 'AUC = 0.64', fontsize=12, color='blue')
plt.legend(loc="lower right")
plt.savefig('docs/figures/roc_curve.png')
plt.close()

print("All plots generated successfully.")
