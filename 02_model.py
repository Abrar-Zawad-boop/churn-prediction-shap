import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import os

# ── 1. LOAD DATA ─────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── 2. CLEAN DATA ─────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# ── 3. ENCODE TARGET ─────────────────────────────────────
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
# ── 4. ENCODE CATEGORICAL FEATURES ───────────────────────
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ── 5. SPLIT FEATURES & TARGET ───────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. TRAIN MODEL ───────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # handles class imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 7. EVALUATE ─────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print('\n=== MODEL PERFORMANCE ===')
print(classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')

# ── 8. SHAP EXPLAINABILITY ───────────────────────────────
print('\nCalculating SHAP values...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot — shows which features matter most globally
shap.summary_plot(
    shap_values[:, :, 1],  # new SHAP format for multi-output
    X_test,
    plot_type='bar',
    show=True
)

# ── 9. SAVE MODEL ────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/churn_model.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')
print('\nModel saved to model/churn_model.pkl')