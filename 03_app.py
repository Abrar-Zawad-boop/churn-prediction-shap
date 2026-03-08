import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title='Churn Predictor',
    page_icon='📊',
    layout='wide'
)

# ── LOAD MODEL ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('model/churn_model.pkl')
    features = joblib.load('model/feature_names.pkl')
    return model, features

model, feature_names = load_model()

# ── HEADER ──────────────────────────────────────────────
st.title('Customer Churn Predictor')
st.markdown('Enter customer details to predict churn risk and understand the reasons why.')
st.divider()

# ── INPUT SIDEBAR ────────────────────────────────────────
st.sidebar.header('Customer Details')

tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
monthly = st.sidebar.slider('Monthly Charges ($)', 20, 120, 65)
total = tenure * monthly
contract = st.sidebar.selectbox('Contract Type', [0, 1, 2],
    format_func=lambda x: ['Month-to-Month', 'One Year', 'Two Year'][x])
internet = st.sidebar.selectbox('Internet Service', [0, 1, 2],
    format_func=lambda x: ['DSL', 'Fiber Optic', 'No'][x])
senior = st.sidebar.selectbox('Senior Citizen', [0, 1],
    format_func=lambda x: 'No' if x == 0 else 'Yes')
tech_support = st.sidebar.selectbox('Tech Support', [0, 1],
    format_func=lambda x: 'No' if x == 0 else 'Yes')

# ── BUILD INPUT ─────────────────────────────────────────
# Build a default row from first row of training data structure
input_dict = {f: 0 for f in feature_names}
input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly
input_dict['TotalCharges'] = total
input_dict['Contract'] = contract
input_dict['InternetService'] = internet
input_dict['SeniorCitizen'] = senior
input_dict['TechSupport'] = tech_support

input_df = pd.DataFrame([input_dict])[feature_names]

# ── PREDICTION ──────────────────────────────────────────
prob = model.predict_proba(input_df)[0][1]
risk_label = 'HIGH RISK' if prob > 0.5 else 'LOW RISK'
color = 'red' if prob > 0.5 else 'green'

col1, col2 = st.columns(2)
with col1:
    st.metric('Churn Probability', f'{prob:.1%}')
    st.markdown(f'### :{color}[{risk_label}]')

with col2:
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=prob * 100,
        title={'text': 'Risk Score'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkred' if prob > 0.5 else 'green'},
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'},
            ]
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# ── SHAP EXPLANATION ─────────────────────────────────────
st.divider()
st.subheader('Why is this customer at risk? (SHAP Explanation)')

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(input_df)

fig2, ax = plt.subplots(figsize=(10, 4))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals[0, :, 1],
        base_values=explainer.expected_value[1],
        data=input_df.iloc[0],
        feature_names=feature_names
    ),
    show=False
)
st.pyplot(fig2)

st.caption('Red bars push prediction toward CHURN. Blue bars push toward STAY.')