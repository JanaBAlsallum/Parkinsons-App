import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf

import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Adam = tf.keras.optimizers.Adam

# Page configuration
st.set_page_config(page_title="Parkinson's Detection", layout="wide")

st.title("🧠 Parkinson's Disease Detection System")
st.markdown("### Enter voice measurements to get the prediction")

# Load and train models
@st.cache_resource
def train_models():
    # Load data
    df = pd.read_csv('parkinsons.data')
    df_processed = df.drop('name', axis=1)
    
    X = df_processed.drop('status', axis=1)
    y = df_processed['status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Train Neural Network
    input_dim = X_train_scaled.shape[1]
    nn_model = Sequential()
    nn_model.add(tf.keras.Input(shape=(input_dim,)))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(BatchNormalization())
    nn_model.add(Dropout(0.3))
    nn_model.add(Dense(16, activation='relu'))
    nn_model.add(BatchNormalization())
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(1, activation='sigmoid'))
    
    nn_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Calculate accuracy
    knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    nn_pred = (nn_model.predict(X_test_scaled).ravel() >= 0.5).astype(int)
    nn_acc = accuracy_score(y_test, nn_pred)
    
    return knn, svm, nn_model, scaler, X_test_scaled, y_test, knn_acc, svm_acc, nn_acc, X.columns.tolist()

# Display model accuracy in sidebar
st.sidebar.header("Model Performance")

try:
    knn, svm, nn_model, scaler, X_test, y_test, knn_acc, svm_acc, nn_acc, feature_names = train_models()
    
    st.sidebar.metric("KNN Accuracy", f"{knn_acc*100:.1f}%")
    st.sidebar.metric("SVM Accuracy", f"{svm_acc*100:.1f}%")
    st.sidebar.metric("Neural Network", f"{nn_acc*100:.1f}%")
    st.sidebar.success(f"Best Model: SVM with {svm_acc*100:.1f}% accuracy")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure the file 'parkinsons.data' is in the same folder")
    st.stop()

# Model selection
st.subheader("Select Model")
model_choice = st.radio(
    "Choose the model to use for prediction:",
    ["SVM (Recommended)", "KNN", "Neural Network"],
    horizontal=True
)

# Data input
st.subheader("Enter Voice Measurements")

col1, col2 = st.columns(2)
user_input = []

for i, feature in enumerate(feature_names):
    if i < 11:
        val = col1.number_input(f"{feature}", value=0.0, format="%.6f", key=f"f_{i}")
    else:
        val = col2.number_input(f"{feature}", value=0.0, format="%.6f", key=f"f_{i}")
    user_input.append(val)

# Prediction button
if st.button("Predict Now", type="primary", use_container_width=True):
    # Convert input
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Predict based on selected model
    if model_choice == "SVM (Recommended)":
        prediction = svm.predict(input_scaled)[0]
        proba = svm.predict_proba(input_scaled)[0]
        model_name = "SVM"
        model_acc = svm_acc
    elif model_choice == "KNN":
        prediction = knn.predict(input_scaled)[0]
        proba = None
        model_name = "KNN"
        model_acc = knn_acc
    else:
        prediction = (nn_model.predict(input_scaled)[0][0] >= 0.5).astype(int)
        prob = nn_model.predict(input_scaled)[0][0]
        proba = [1-prob, prob]
        model_name = "Neural Network"
        model_acc = nn_acc
    
    st.markdown("---")
    
    # Display result
    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    with col_res2:
        if prediction == 1:
            st.error("**Result: Parkinson's Disease Detected**")
        else:
            st.success("**Result: Healthy - No Parkinson's Disease Detected**")
        
        if proba is not None:
            st.progress(float(proba[1]))
            st.caption(f"Confidence: {proba[1]*100:.1f}%")
        
        st.caption(f"📊 Model used: {model_name} | Model accuracy: {model_acc*100:.1f}%")

# Display model comparison
with st.expander("📈 Model Performance Comparison"):
    comparison_df = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'Neural Network'],
        'Accuracy (%)': [f"{knn_acc*100:.1f}", f"{svm_acc*100:.1f}", f"{nn_acc*100:.1f}"]
    })
    st.table(comparison_df)

st.markdown("---")
st.caption("This system is for research and educational purposes only")