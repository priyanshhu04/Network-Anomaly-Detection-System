import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class NetworkAnomalyDetector:
    """Simple Network Anomaly Detection System"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.iso_model = None
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """Load KDD Cup 1999 dataset"""
        try:
            # Column names for KDD dataset
            columns = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
            ]
            
            # Load dataset from URL
            url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
            df = pd.read_csv(url, header=None, names=columns)
            self.data = df
            return df
            
        except Exception as e:
            # If internet fails, create sample data
            print(f"Could not download dataset: {e}. Using sample data.")
            np.random.seed(42)
            sample_data = {
        'duration': np.random.randint(0, 1000, 5000),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], 5000),
        'service': np.random.choice(['http', 'ftp', 'smtp'], 5000),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], 5000),
        'src_bytes': np.random.randint(0, 10000, 5000),
        'dst_bytes': np.random.randint(0, 10000, 5000),
        'land': np.zeros(5000),
        'wrong_fragment': np.zeros(5000),
        'urgent': np.zeros(5000),
        'hot': np.zeros(5000),
        'num_failed_logins': np.zeros(5000),
        'logged_in': np.ones(5000),
        'num_compromised': np.zeros(5000),
        'root_shell': np.zeros(5000),
        'su_attempted': np.zeros(5000),
        'num_root': np.zeros(5000),
        'num_file_creations': np.zeros(5000),
        'num_shells': np.zeros(5000),
        'num_access_files': np.zeros(5000),
        'num_outbound_cmds': np.zeros(5000),
        'is_host_login': np.zeros(5000),
        'is_guest_login': np.zeros(5000),
        'count': np.random.randint(1, 100, 5000),
        'srv_count': np.random.randint(1, 50, 5000),
        'serror_rate': np.random.rand(5000),
        'srv_serror_rate': np.random.rand(5000),
        'rerror_rate': np.random.rand(5000),
        'srv_rerror_rate': np.random.rand(5000),
        'same_srv_rate': np.random.rand(5000),
        'diff_srv_rate': np.random.rand(5000),
        'srv_diff_host_rate': np.random.rand(5000),
        'dst_host_count': np.random.randint(1, 255, 5000),
        'dst_host_srv_count': np.random.randint(1, 255, 5000),
        'dst_host_same_srv_rate': np.random.rand(5000),
        'dst_host_diff_srv_rate': np.random.rand(5000),
        'dst_host_same_src_port_rate': np.random.rand(5000),
        'dst_host_srv_diff_host_rate': np.random.rand(5000),
        'dst_host_serror_rate': np.random.rand(5000),
        'dst_host_srv_serror_rate': np.random.rand(5000),
        'dst_host_rerror_rate': np.random.rand(5000),
        'dst_host_srv_rerror_rate': np.random.rand(5000),
        'label': np.random.choice(['normal.', 'neptune.', 'smurf.', 'back.'], 5000, p=[0.6, 0.2, 0.1, 0.1])
            }
            df = pd.DataFrame(sample_data)
            self.data = df
            return df

    def preprocess_data(self, df):
        """Simple preprocessing for the dataset"""
        # Create binary labels: 0 = normal, 1 = attack
        df = df.copy()
        df['is_attack'] = (df['class'] != 'normal.').astype(int)
        
        # Select key features for simplicity
        key_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'protocol_type', 'service', 'flag']
        
        # Keep only available features
        available_features = [col for col in key_features if col in df.columns]
        df_subset = df[available_features + ['is_attack', 'class']].copy()
        
        # Encode categorical variables
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in df_subset.columns:
                le = LabelEncoder()
                df_subset[col] = le.fit_transform(df_subset[col].astype(str))
        
        # Handle missing values
        df_subset = df_subset.fillna(0)
        self.processed_data = df_subset
        return df_subset

    def train_isolation_forest(self, X, contamination=0.1):
        """Train Isolation Forest model"""
        self.iso_model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        self.iso_model.fit(X)
        return self.iso_model

    def simple_autoencoder_anomaly(self, X, threshold_percentile=95):
        """Simple autoencoder-like anomaly detection using reconstruction error"""
        # Calculate mean and std for each feature
        mean_vals = np.mean(X, axis=0)
        std_vals = np.std(X, axis=0)
        
        # Calculate reconstruction error (simplified version)
        reconstruction_errors = np.sum(((X - mean_vals) / (std_vals + 1e-8)) ** 2, axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        return predictions, reconstruction_errors, threshold

    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }

    def get_data_statistics(self):
        """Get basic statistics about the dataset"""
        if self.data is None or self.processed_data is None:
            return None
            
        stats = {
            'total_records': len(self.data),
            'attack_records': self.processed_data['is_attack'].sum(),
            'normal_records': (self.processed_data['is_attack'] == 0).sum(),
            'features_count': len([col for col in self.processed_data.columns if col not in ['is_attack', 'class']]),
            'attack_types': self.data['class'].value_counts().to_dict()
        }
        return stats

    def create_attack_distribution_plot(self):
        """Create attack distribution pie chart"""
        if self.data is None:
            return None
            
        attack_counts = self.data['class'].value_counts().head(10)
        fig = px.pie(values=attack_counts.values, names=attack_counts.index, 
                    title="Distribution of Traffic Types")
        return fig

    def create_normal_vs_attack_plot(self):
        """Create normal vs attack bar chart"""
        if self.processed_data is None:
            return None
            
        normal_attack = self.processed_data['is_attack'].value_counts()
        fig = px.bar(x=['Normal', 'Attack'], y=normal_attack.values,
                    title="Normal Traffic vs Attacks", color=['Normal', 'Attack'])
        return fig

    def create_feature_histogram(self, feature_name):
        """Create histogram for a specific feature"""
        if self.processed_data is None or feature_name not in self.processed_data.columns:
            return None
            
        fig = px.histogram(self.processed_data, x=feature_name, color='is_attack',
                         title=f"Distribution of {feature_name} by Traffic Type",
                         labels={'is_attack': 'Traffic Type'})
        return fig

    def create_performance_radar_chart(self, iso_metrics, ae_metrics):
        """Create radar chart comparing model performance"""
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig.add_trace(go.Scatterpolar(
            r=[iso_metrics['Accuracy'], iso_metrics['Precision'], 
               iso_metrics['Recall'], iso_metrics['F1-Score']],
            theta=metrics,
            fill='toself',
            name='Isolation Forest'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[ae_metrics['Accuracy'], ae_metrics['Precision'], 
               ae_metrics['Recall'], ae_metrics['F1-Score']],
            theta=metrics,
            fill='toself',
            name='Autoencoder-style'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        return fig

    def create_confusion_matrix_plot(self, y_true, y_pred, title):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual"),
                       title=title)
        return fig

    def create_reconstruction_error_plot(self, reconstruction_errors, threshold):
        """Create reconstruction error histogram"""
        fig = px.histogram(x=reconstruction_errors, nbins=50,
                         title="Distribution of Reconstruction Errors")
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                    annotation_text="Threshold")
        return fig
