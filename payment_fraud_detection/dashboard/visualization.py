import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_distribution_plot(df, column, title):
    """Create distribution plot"""
    if column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            # Numerical distribution
            fig = px.histogram(
                df, x=column, title=title,
                color_discrete_sequence=['#1E88E5']
            )
        else:
            # Categorical distribution
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            fig = px.bar(
                value_counts, x=column, y='count',
                title=title, color_discrete_sequence=['#1E88E5']
            )
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title=column,
            yaxis_title='Count'
        )
        
        return fig
    else:
        return go.Figure()

def create_correlation_heatmap(df, numeric_cols):
    """Create correlation heatmap"""
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            title_x=0.5,
            title_font_size=16,
            width=600,
            height=600
        )
        
        return fig
    else:
        return go.Figure()

def create_fraud_by_category(df, category_col, fraud_col='fraud'):
    """Create fraud rate by category plot"""
    if category_col in df.columns and fraud_col in df.columns:
        fraud_rate = df.groupby(category_col)[fraud_col].mean().sort_values(ascending=False).head(15).reset_index()
        fraud_rate.columns = [category_col, 'fraud_rate']
        
        fig = px.bar(
            fraud_rate, x='fraud_rate', y=category_col,
            title=f'Fraud Rate by {category_col}',
            orientation='h',
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='Fraud Rate',
            yaxis_title=category_col,
            yaxis={'categoryorder':'total ascending'}
        )
        
        return fig
    else:
        return go.Figure()

def create_amount_distribution(df, fraud_col='fraud'):
    """Create amount distribution by fraud status"""
    if 'amount' in df.columns and fraud_col in df.columns:
        fig = px.box(
            df, x=fraud_col, y='amount',
            title='Transaction Amount Distribution by Fraud Status',
            color=fraud_col,
            color_discrete_map={0: '#1E88E5', 1: '#EF5350'}
        )
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            xaxis_title='Fraud Status (0=Legitimate, 1=Fraud)',
            yaxis_title='Amount'
        )
        
        return fig
    else:
        return go.Figure()