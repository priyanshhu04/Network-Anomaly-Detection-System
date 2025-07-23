# app.py - Streamlit Web Application

import streamlit as st
import pandas as pd
import numpy as np
from main import NetworkAnomalyDetector
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="🔒 Network Anomaly Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Initialize the detector
@st.cache_resource
def initialize_detector():
    detector = NetworkAnomalyDetector()
    return detector

@st.cache_data
def load_and_process_data():
    detector = initialize_detector()
    df = detector.load_data()
    df_processed = detector.preprocess_data(df)
    return detector, df, df_processed

def main():
    st.markdown('<h1 class="main-header">🔒 Network Anomaly Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox("Choose a section:", 
                               ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "📈 Results"])
    
    # Load data
    with st.spinner("Loading dataset..."):
        detector, df, df_processed = load_and_process_data()
        stats = detector.get_data_statistics()
    
    if page == "🏠 Home":
        show_home_page(stats)
    elif page == "📊 Data Analysis":
        show_data_analysis_page(detector, df, df_processed)
    elif page == "🤖 Model Training":
        show_model_training_page(detector, df_processed)
    elif page == "📈 Results":
        show_results_page(detector, df_processed)

def show_home_page(stats):
    st.markdown("""
    ## Welcome to the Network Anomaly Detection Dashboard! 🎯
    
    This application detects network anomalies and potential cyber attacks using machine learning.
    
    ### 🔍 What This Project Does:
    - **Analyzes network traffic data** from the famous KDD Cup 1999 dataset
    - **Detects anomalies** using two powerful algorithms:
      - 🌲 **Isolation Forest**: Finds unusual patterns by isolating anomalies
      - 🧠 **Autoencoder-style Detection**: Identifies patterns that don't fit normal behavior
    - **Provides insights** through interactive visualizations
    
    ### 📈 Key Statistics:
    """)
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Records", f"{stats['total_records']:,}")
        with col2:
            st.metric("⚠️ Attack Records", f"{stats['attack_records']:,}")
        with col3:
            st.metric("✅ Normal Records", f"{stats['normal_records']:,}")
        with col4:
            st.metric("📊 Features Used", stats['features_count'])

def show_data_analysis_page(detector, df, df_processed):
    st.header("📊 Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attack Distribution")
        fig = detector.create_attack_distribution_plot()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Normal vs Attack")
        fig = detector.create_normal_vs_attack_plot()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Analysis")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_attack']]
    
    if len(numeric_cols) > 0:
        selected_feature = st.selectbox("Select feature to analyze:", numeric_cols)
        fig = detector.create_feature_histogram(selected_feature)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df_processed.head(10))

def show_model_training_page(detector, df_processed):
    st.header("🤖 Anomaly Detection Models")
    
    # Prepare data for models
    feature_cols = [col for col in df_processed.columns if col not in ['is_attack', 'class']]
    X = df_processed[feature_cols].values
    y = df_processed['is_attack'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.subheader("🌲 Isolation Forest Results")
    
    # Train Isolation Forest
    contamination = st.slider("Contamination Rate (Expected % of anomalies):", 0.05, 0.3, 0.1, 0.01)
    
    if st.button("🚀 Train Isolation Forest"):
        with st.spinner("Training Isolation Forest..."):
            iso_model = detector.train_isolation_forest(X_scaled, contamination=contamination)
            iso_predictions = iso_model.predict(X_scaled)
            iso_predictions = (iso_predictions == -1).astype(int)  # Convert to 0/1
            
            # Calculate metrics
            iso_metrics = detector.calculate_metrics(y, iso_predictions)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{iso_metrics['Accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{iso_metrics['Precision']:.3f}")
            with col3:
                st.metric("Recall", f"{iso_metrics['Recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{iso_metrics['F1-Score']:.3f}")
    
    st.subheader("🧠 Simplified Autoencoder Results")
    
    threshold_percentile = st.slider("Detection Threshold (Percentile):", 90, 99, 95, 1)
    
    if st.button("🚀 Train Autoencoder-style Detector"):
        with st.spinner("Training Autoencoder-style detector..."):
            ae_predictions, reconstruction_errors, threshold = detector.simple_autoencoder_anomaly(
                X_scaled, threshold_percentile=threshold_percentile)
            
            # Calculate metrics
            ae_metrics = detector.calculate_metrics(y, ae_predictions)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{ae_metrics['Accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{ae_metrics['Precision']:.3f}")
            with col3:
                st.metric("Recall", f"{ae_metrics['Recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{ae_metrics['F1-Score']:.3f}")
            
            # Plot reconstruction errors
            fig = detector.create_reconstruction_error_plot(reconstruction_errors, threshold)
            st.plotly_chart(fig, use_container_width=True)

def show_results_page(detector, df_processed):
    st.header("📈 Model Performance Summary")
    
    # Prepare data
    feature_cols = [col for col in df_processed.columns if col not in ['is_attack', 'class']]
    X = df_processed[feature_cols].values
    y = df_processed['is_attack'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train both models
    with st.spinner("Training models for comparison..."):
        # Isolation Forest
        iso_model = detector.train_isolation_forest(X_scaled, contamination=0.1)
        iso_predictions = iso_model.predict(X_scaled)
        iso_predictions = (iso_predictions == -1).astype(int)
        iso_metrics = detector.calculate_metrics(y, iso_predictions)
        
        # Autoencoder-style
        ae_predictions, _, _ = detector.simple_autoencoder_anomaly(X_scaled, threshold_percentile=95)
        ae_metrics = detector.calculate_metrics(y, ae_predictions)
    
    # Create comparison
    comparison_df = pd.DataFrame({
        'Model': ['Isolation Forest', 'Autoencoder-style'],
        'Accuracy': [iso_metrics['Accuracy'], ae_metrics['Accuracy']],
        'Precision': [iso_metrics['Precision'], ae_metrics['Precision']],
        'Recall': [iso_metrics['Recall'], ae_metrics['Recall']],
        'F1-Score': [iso_metrics['F1-Score'], ae_metrics['F1-Score']]
    })
    
    st.subheader("🏆 Model Comparison")
    st.dataframe(comparison_df)
    
    # Visualization
    fig = detector.create_performance_radar_chart(iso_metrics, ae_metrics)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Isolation Forest Confusion Matrix")
        fig = detector.create_confusion_matrix_plot(y, iso_predictions, "Isolation Forest")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Autoencoder-style Confusion Matrix")
        fig = detector.create_confusion_matrix_plot(y, ae_predictions, "Autoencoder-style")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
