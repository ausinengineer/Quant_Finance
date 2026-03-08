import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_sidebar():
    """Render sidebar with navigation and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=100)
        st.title("Fraud Detection")
        st.markdown("---")
        
        st.subheader("About")
        st.info(
            """
            This application uses a deep learning model (PyTorch) 
            to detect fraudulent payment transactions in real-time.
            
            **Features:**
            - Data exploration
            - Model training
            - Model evaluation
            - Real-time prediction
            """
        )
        
        st.markdown("---")
        
        # Model status
        st.subheader("System Status")
        
        import os
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')] if os.path.exists('models') else []
        
        if model_files:
            st.success(f"✅ {len(model_files)} trained model(s) available")
            latest_model = sorted(model_files)[-1] if model_files else None
            if latest_model:
                st.caption(f"Latest: {latest_model[:20]}...")
        else:
            st.warning("⚠️ No trained models found")
        
        st.markdown("---")
        st.caption("© 2024 Fraud Detection System")

def render_metrics(metrics):
    """Render evaluation metrics in cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy",
            f"{metrics.get('accuracy', 0):.4f}",
            help="Proportion of correct predictions"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{metrics.get('precision', 0):.4f}",
            help="Proportion of true fraud among predicted fraud"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{metrics.get('recall', 0):.4f}",
            help="Proportion of fraud cases correctly identified"
        )
    
    with col4:
        st.metric(
            "F1 Score",
            f"{metrics.get('f1_score', 0):.4f}",
            help="Harmonic mean of precision and recall"
        )
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics.get('roc_auc', 0):.4f}",
            help="Area under ROC curve"
        )
    
    with col6:
        st.metric(
            "Avg Precision",
            f"{metrics.get('average_precision', 0):.4f}",
            help="Average precision score"
        )

def render_predictions(result):
    """Render prediction results"""
    st.subheader("Prediction Result")
    
    # Color based on risk
    if result['risk_level'] == 'High':
        bg_color = "#ffebee"
        border_color = "#ef5350"
    elif result['risk_level'] == 'Medium':
        bg_color = "#fff3e0"
        border_color = "#ff9800"
    else:
        bg_color = "#e8f5e8"
        border_color = "#66bb6a"
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 5px solid {border_color};
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    ">
        <h3 style="margin: 0; color: {border_color};">
            {result['risk_level']} RISK
        </h3>
        <p style="font-size: 24px; margin: 10px 0;">
            Fraud Probability: {result['fraud_probability']:.2%}
        </p>
        <p style="font-size: 18px; margin: 0;">
            Prediction: <strong>{'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['fraud_probability'] * 100,
        title={'text': "Fraud Risk Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': result['fraud_probability'] * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)