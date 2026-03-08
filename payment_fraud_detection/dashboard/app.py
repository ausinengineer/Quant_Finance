import streamlit as st
import pandas as pd
import numpy as np
import torch
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.data_preprocessing import DataPreprocessor
from src.predict import Predictor
from src.evaluate import Evaluator
from src.utils import load_model, load_metrics
from dashboard.components import render_sidebar, render_metrics, render_predictions
from dashboard.visualization import create_distribution_plot, create_correlation_heatmap

# Page configuration
st.set_page_config(
    page_title="Payment Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fraud-highlight {
        color: #dc3545;
        font-weight: 700;
    }
    .legit-highlight {
        color: #28a745;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'preprocessor_loaded' not in st.session_state:
    st.session_state.preprocessor_loaded = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Payment Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown("Deep Learning-based fraud detection using PyTorch")
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Explorer", 
        "🤖 Model Training", 
        "📈 Model Evaluation", 
        "🔮 Real-time Prediction"
    ])
    
    with tab1:
        render_data_explorer()
    
    with tab2:
        render_model_training()
    
    with tab3:
        render_model_evaluation()
    
    with tab4:
        render_prediction_interface()

def render_data_explorer():
    """Data exploration tab"""
    st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show data sample
            st.subheader("Data Sample")
            st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        if st.session_state.data_loaded:
            df = st.session_state.df
            
            # Data statistics
            st.subheader("Data Statistics")
            
            fraud_count = df['fraud'].sum() if 'fraud' in df.columns else 0
            fraud_rate = fraud_count / len(df) * 100 if 'fraud' in df.columns else 0
            
            st.metric("Total Transactions", f"{len(df):,}")
            if 'fraud' in df.columns:
                st.metric("Fraud Cases", f"{int(fraud_count):,}", 
                         delta=f"{fraud_rate:.2f}%", delta_color="inverse")
            
            st.metric("Unique Customers", df['customer'].nunique() if 'customer' in df.columns else 0)
            st.metric("Unique Merchants", df['merchant'].nunique() if 'merchant' in df.columns else 0)
    
    # Visualizations
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution
            if 'fraud' in df.columns:
                fig = create_distribution_plot(df, 'fraud', 'Fraud Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Category distribution
            if 'category' in df.columns:
                fig = px.bar(
                    df['category'].value_counts().head(10).reset_index(),
                    x='count', y='category',
                    title='Top 10 Transaction Categories',
                    orientation='h',
                    color_discrete_sequence=['#1E88E5']
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Amount distribution
            if 'amount' in df.columns:
                fig = px.histogram(
                    df, x='amount', nbins=50,
                    title='Transaction Amount Distribution',
                    color_discrete_sequence=['#1E88E5']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Gender distribution
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title='Gender Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)

def render_model_training():
    """Model training tab"""
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the Data Explorer tab.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox("Model Architecture", ['basic', 'advanced'], index=0)
        hidden_layers = st.text_input("Hidden Layers (comma-separated)", "256,128,64")
    
    with col2:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0)
    
    with col3:
        epochs = st.number_input("Epochs", min_value=5, max_value=200, value=30, step=5)
        dropout_rate = st.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    
    use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training in progress... This may take a few minutes."):
            try:
                # Parse hidden layers
                hidden_dims = [int(x.strip()) for x in hidden_layers.split(',')]
                
                # Train model
                result = train_model(
                    st.session_state.df,
                    model_type=model_type,
                    hidden_dims=hidden_dims,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    dropout_rate=dropout_rate,
                    use_smote=use_smote
                )
                
                st.success("✅ Model training completed!")
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Test Accuracy", f"{result['test_accuracy']:.4f}")
                with col2:
                    st.metric("Test Precision", f"{result['test_precision']:.4f}")
                with col3:
                    st.metric("Test Recall", f"{result['test_recall']:.4f}")
                with col4:
                    st.metric("Test F1 Score", f"{result['test_f1']:.4f}")
                
                # Show training plot
                if os.path.exists('graphs/training_history.png'):
                    st.image('graphs/training_history.png', use_column_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def render_model_evaluation():
    """Model evaluation tab"""
    st.markdown('<h2 class="sub-header">Model Evaluation</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')] if os.path.exists('models') else []
    
    if not model_files:
        st.warning("No trained models found. Please train a model first.")
        return
    
    selected_model = st.selectbox("Select Model", model_files)
    
    if st.button("🔍 Evaluate Model", use_container_width=True):
        with st.spinner("Evaluating model..."):
            try:
                # Load model and evaluate
                result = evaluate_model(selected_model, st.session_state.df)
                
                # Display metrics
                render_metrics(result['metrics'])
                
                # Show plots
                col1, col2 = st.columns(2)
                
                with col1:
                    if os.path.exists('graphs/confusion_matrix.png'):
                        st.image('graphs/confusion_matrix.png', use_column_width=True)
                
                with col2:
                    if os.path.exists('graphs/roc_curve.png'):
                        st.image('graphs/roc_curve.png', use_column_width=True)
                
                if os.path.exists('graphs/precision_recall_curve.png'):
                    st.image('graphs/precision_recall_curve.png', use_column_width=True)
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

def render_prediction_interface():
    """Real-time prediction tab"""
    st.markdown('<h2 class="sub-header">Real-time Fraud Prediction</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')] if os.path.exists('models') else []
    
    if not model_files:
        st.warning("No trained models found. Please train a model first.")
        return
    
    selected_model = st.selectbox("Select Model for Prediction", model_files, key='pred_model')
    
    tab_single, tab_batch = st.tabs(["Single Transaction", "Batch Prediction"])
    
    with tab_single:
        st.subheader("Single Transaction Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer = st.text_input("Customer ID", "C12345")
            age = st.selectbox("Age Group", ['1', '2', '3', '4', '5', '6'])
            gender = st.selectbox("Gender", ['M', 'F'])
            zipcode = st.text_input("Zip Code", "28007")
        
        with col2:
            merchant = st.text_input("Merchant ID", "M348934600")
            merchant_zip = st.text_input("Merchant Zip", "28007")
            category = st.selectbox("Category", [
                'es_transportation', 'es_health', 'es_food', 'es_hotelservices',
                'es_otherservices', 'es_fashion', 'es_tech', 'es_sportsandtoys',
                'es_home', 'es_barsandrestaurants', 'es_hyper', 'es_travel',
                'es_wellnessandbeauty', 'es_contents', 'es_leisure'
            ])
            amount = st.number_input("Amount", min_value=0.01, value=100.0, step=10.0)
        
        if st.button("🔍 Predict Fraud", key='predict_single'):
            transaction = {
                'customer': customer,
                'age': age,
                'gender': gender,
                'zipcodeOri': zipcode,
                'merchant': merchant,
                'zipMerchant': merchant_zip,
                'category': category,
                'amount': amount
            }
            
            result = predict_single(selected_model, transaction)
            render_predictions(result)
    
    with tab_batch:
        st.subheader("Batch Prediction")
        
        batch_file = st.file_uploader("Upload transactions CSV for batch prediction", type=['csv'])
        
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.write(f"Loaded {len(batch_df)} transactions for prediction")
            st.dataframe(batch_df.head())
            
            if st.button("🔍 Run Batch Prediction", key='predict_batch'):
                results = predict_batch(selected_model, batch_df)
                st.session_state.predictions = results
                
                st.success(f"Predictions completed for {len(results)} transactions")
                
                # Summary
                fraud_count = results['predicted_fraud'].sum()
                st.metric("Predicted Fraud Cases", int(fraud_count))
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(results)

def train_model(df, model_type='basic', hidden_dims=[256, 128, 64], 
                batch_size=64, learning_rate=0.001, epochs=30, dropout_rate=0.3, use_smote=True):
    """Train model (simplified for dashboard)"""
    from src.train import Trainer
    from src.evaluate import Evaluator
    from src.utils import set_seed, print_model_summary, get_timestamp
    
    set_seed(42)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df, fit_scaler=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, use_smote=use_smote
    )
    
    # Create model
    input_dim = X_train.shape[1]
    model = get_model(model_type, input_dim, hidden_dims, dropout_rate)
    print_model_summary(model, input_dim)
    
    # Train
    trainer = Trainer(model)
    trainer.set_optimizer('adam', lr=learning_rate)
    trainer.set_class_weights(train_loader)
    
    timestamp = get_timestamp()
    model_path = f'models/fraud_model_{timestamp}.pth'
    
    trainer.train(train_loader, val_loader, epochs=epochs, save_path=model_path)
    trainer.plot_training_history()
    
    # Save preprocessor
    preprocessor.save_preprocessor(f'models/preprocessor_{timestamp}.pkl')
    
    # Evaluate
    evaluator = Evaluator(model)
    y_test, y_pred, y_probs = evaluator.evaluate(test_loader)
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_probs)
    
    evaluator.print_classification_report(y_test, y_pred)
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_probs)
    evaluator.plot_precision_recall_curve(y_test, y_probs)
    
    result = {
        'test_accuracy': metrics['accuracy'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'test_f1': metrics['f1_score'],
        'test_roc_auc': metrics['roc_auc'],
        'model_path': model_path
    }
    
    return result

def evaluate_model(model_filename, df):
    """Evaluate model (simplified for dashboard)"""
    from src.model import get_model
    from src.data_preprocessing import DataPreprocessor
    from src.evaluate import Evaluator
    import joblib
    
    # Load preprocessor
    preprocessor_path = model_filename.replace('.pth', '.pkl').replace('models/', 'models/preprocessor_')
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)
    
    # Prepare test data
    X, y = preprocessor.prepare_data(df, fit_scaler=False)
    
    # Use only test split (simplified)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create test loader
    from src.data_preprocessing import FraudDataset
    from torch.utils.data import DataLoader
    
    test_dataset = FraudDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    input_dim = X_test.shape[1]
    model = get_model('basic', input_dim)
    model_path = os.path.join('models', model_filename)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluator = Evaluator(model)
    y_test, y_pred, y_probs = evaluator.evaluate(test_loader)
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_probs)
    
    # Generate plots
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_probs)
    evaluator.plot_precision_recall_curve(y_test, y_probs)
    
    return {'metrics': metrics}

def predict_single(model_filename, transaction):
    """Predict single transaction"""
    from src.model import get_model
    from src.data_preprocessing import DataPreprocessor
    from src.predict import Predictor
    import joblib
    
    # Load preprocessor
    preprocessor_path = model_filename.replace('.pth', '.pkl').replace('models/', 'models/preprocessor_')
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)
    
    # Load model
    # Need to get input_dim from preprocessor
    input_dim = len(preprocessor.feature_columns)
    model = get_model('basic', input_dim)
    model_path = os.path.join('models', model_filename)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Predict
    predictor = Predictor(model, preprocessor)
    result = predictor.predict_single(transaction)
    
    return result

def predict_batch(model_filename, df):
    """Predict batch transactions"""
    from src.model import get_model
    from src.data_preprocessing import DataPreprocessor
    from src.predict import Predictor
    
    # Load preprocessor
    preprocessor_path = model_filename.replace('.pth', '.pkl').replace('models/', 'models/preprocessor_')
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)
    
    # Load model
    input_dim = len(preprocessor.feature_columns)
    model = get_model('basic', input_dim)
    model_path = os.path.join('models', model_filename)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Predict
    predictor = Predictor(model, preprocessor)
    results = predictor.predict_batch(df)
    
    return results

if __name__ == "__main__":
    main()