
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Credit Risk Prediction App")

@st.cache_data
def preprocess_data(df, target_col):
    df = df.copy()
    df.replace('?', np.nan, inplace=True)
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # Encode target
    y = LabelEncoder().fit_transform(y.astype(str))

    # Encode categorical features and impute missing values
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('missing'))
        else:
            imputer = SimpleImputer(strategy='mean')
            X[col] = imputer.fit_transform(X[[col]])

    return X, y

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)  # Keep all class probabilities
        
        # Check if multiclass or binary
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        else:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    return results


uploaded_file = st.file_uploader("Upload your credit dataset CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the target column", df.columns)

    if st.button("Run ML Pipeline"):
        with st.spinner('Processing and training...'):
            X, y = preprocess_data(df, target_col)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = train_models(X_train, y_train)
            results = evaluate_models(models, X_test, y_test)

        st.success("Training completed!")

        st.subheader("Model Evaluation Results")
        for model_name, metrics in results.items():
            st.markdown(f"### {model_name}")
            st.write(f"Accuracy: {metrics['accuracy']:.3f}")
            st.write(f"ROC-AUC: {metrics['roc_auc']:.3f}")
            st.text("Classification Report:")
            st.json(metrics['report'])

        st.subheader("Sample Prediction")
        sample_data = {}
        for col in X.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_vals = df[col].dropna().unique()
                sample_data[col] = st.selectbox(f"Select value for {col}", unique_vals)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                sample_data[col] = st.slider(f"Select value for {col}", min_val, max_val, mean_val)

        if st.button("Predict Loan Approval"):
            input_df = pd.DataFrame([sample_data])
            # Preprocess input similarly
            for col in input_df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    le = LabelEncoder()
                    le.fit(df[col].astype(str).fillna('missing'))
                    input_df[col] = le.transform(input_df[col].astype(str))
                else:
                    input_df[col] = input_df[col].astype(float)

            pred = models['Random Forest'].predict(input_df)[0]
            proba = models['Random Forest'].predict_proba(input_df)[0][1]

            st.write(f"Prediction: {'Approved' if pred == 1 else 'Not Approved'}")
            st.write(f"Approval Probability: {proba:.2f}")

else:
    st.info("Upload a CSV file to get started.")
