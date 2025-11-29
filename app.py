"""
Streamlit Dashboard for Hotel Booking Cancellation Prediction Pipeline
Interactive web interface to explore data, view model performance, and make predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import config

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏨 Hotel Booking Cancellation Prediction Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Hotel+Analytics", width=300)
    st.markdown("### 📊 Navigation")
    
    # Check if data and models exist
    data_exists = config.RAW_DATA_FILE.exists()
    model_exists = config.BEST_MODEL_FILE.exists()
    
    st.markdown(f"""
    **Status:**
    - Data: {'✅ Loaded' if data_exists else '❌ Not Found'}
    - Model: {'✅ Trained' if model_exists else '❌ Not Trained'}
    """)
    
    if not data_exists:
        st.warning("⚠️ No data found! Run: `python generate_sample_data.py`")
    
    if not model_exists:
        st.warning("⚠️ No model found! Run: `python main.py --mode full --quick`")

# Load data
@st.cache_data
def load_data():
    """Load the hotel bookings dataset."""
    if config.RAW_DATA_FILE.exists():
        return pd.read_csv(config.RAW_DATA_FILE)
    return None

@st.cache_resource
def load_model():
    """Load the trained model."""
    if config.BEST_MODEL_FILE.exists():
        return joblib.load(config.BEST_MODEL_FILE)
    return None

df = load_data()
model = load_model()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "🔍 Data Explorer", 
    "🤖 Model Performance", 
    "🎯 Make Predictions",
    "📊 Feature Importance"
])

# Tab 1: Overview
with tab1:
    st.header("Dataset Overview")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bookings", f"{len(df):,}")
        with col2:
            cancellation_rate = df['is_canceled'].mean() * 100 if 'is_canceled' in df.columns else 0
            st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
        with col3:
            st.metric("Features", len(df.columns))
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        st.markdown("---")
        
        # Cancellation distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cancellation Distribution")
            if 'is_canceled' in df.columns:
                fig = px.pie(
                    df, 
                    names='is_canceled',
                    title='Booking Status',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                    labels={'is_canceled': 'Status'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Bookings by Hotel Type")
            if 'hotel' in df.columns:
                hotel_counts = df['hotel'].value_counts()
                fig = px.bar(
                    x=hotel_counts.index,
                    y=hotel_counts.values,
                    labels={'x': 'Hotel Type', 'y': 'Number of Bookings'},
                    title='Distribution by Hotel Type',
                    color=hotel_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Key statistics
        st.subheader("Key Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'lead_time' in df.columns:
                st.markdown(f"""
                **Booking Lead Time:**
                - Average: {df['lead_time'].mean():.0f} days
                - Median: {df['lead_time'].median():.0f} days
                - Max: {df['lead_time'].max():.0f} days
                """)
        
        with col2:
            if 'adr' in df.columns:
                st.markdown(f"""
                **Average Daily Rate (ADR):**
                - Average: ${df['adr'].mean():.2f}
                - Median: ${df['adr'].median():.2f}
                - Max: ${df['adr'].max():.2f}
                """)
    else:
        st.error("❌ No data loaded. Please generate sample data first.")
        st.code("python generate_sample_data.py --samples 10000")

# Tab 2: Data Explorer
with tab2:
    st.header("Data Explorer")
    
    if df is not None:
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'hotel' in df.columns:
                hotel_filter = st.multiselect(
                    "Hotel Type",
                    options=df['hotel'].unique(),
                    default=df['hotel'].unique()
                )
            else:
                hotel_filter = None
        
        with col2:
            if 'is_canceled' in df.columns:
                status_filter = st.multiselect(
                    "Booking Status",
                    options=[0, 1],
                    default=[0, 1],
                    format_func=lambda x: "Not Canceled" if x == 0 else "Canceled"
                )
            else:
                status_filter = None
        
        with col3:
            if 'market_segment' in df.columns:
                segment_filter = st.multiselect(
                    "Market Segment",
                    options=df['market_segment'].unique(),
                    default=df['market_segment'].unique()
                )
            else:
                segment_filter = None
        
        # Apply filters
        filtered_df = df.copy()
        if hotel_filter and 'hotel' in df.columns:
            filtered_df = filtered_df[filtered_df['hotel'].isin(hotel_filter)]
        if status_filter is not None and 'is_canceled' in df.columns:
            filtered_df = filtered_df[filtered_df['is_canceled'].isin(status_filter)]
        if segment_filter and 'market_segment' in df.columns:
            filtered_df = filtered_df[filtered_df['market_segment'].isin(segment_filter)]
        
        st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} bookings**")
        
        # Display data
        st.dataframe(
            filtered_df.head(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_bookings.csv",
            mime="text/csv"
        )
        
        # Visualizations
        st.markdown("---")
        st.subheader("Data Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            if 'lead_time' in filtered_df.columns and 'is_canceled' in filtered_df.columns:
                fig = px.box(
                    filtered_df,
                    x='is_canceled',
                    y='lead_time',
                    title='Lead Time by Cancellation Status',
                    labels={'is_canceled': 'Canceled', 'lead_time': 'Lead Time (days)'},
                    color='is_canceled',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            if 'adr' in filtered_df.columns and 'is_canceled' in filtered_df.columns:
                fig = px.box(
                    filtered_df,
                    x='is_canceled',
                    y='adr',
                    title='ADR by Cancellation Status',
                    labels={'is_canceled': 'Canceled', 'adr': 'Average Daily Rate ($)'},
                    color='is_canceled',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No data available")

# Tab 3: Model Performance
with tab3:
    st.header("Model Performance")
    
    # Check if evaluation file exists
    if config.MODEL_EVALUATION_FILE.exists():
        with open(config.MODEL_EVALUATION_FILE, 'r', encoding='utf-8') as f:
            eval_report = f.read()
        
        # Parse metrics
        lines = eval_report.split('\n')
        metrics = {}
        for line in lines:
            if 'ACCURACY:' in line:
                metrics['Accuracy'] = float(line.split(':')[1].strip())
            elif 'PRECISION:' in line:
                metrics['Precision'] = float(line.split(':')[1].strip())
            elif 'RECALL:' in line:
                metrics['Recall'] = float(line.split(':')[1].strip())
            elif 'F1:' in line:
                metrics['F1-Score'] = float(line.split(':')[1].strip())
            elif 'ROC_AUC:' in line:
                metrics['ROC-AUC'] = float(line.split(':')[1].strip())
        
        # Display metrics
        if metrics:
            st.subheader("Performance Metrics")
            cols = st.columns(len(metrics))
            for idx, (metric, value) in enumerate(metrics.items()):
                with cols[idx]:
                    st.metric(metric, f"{value:.4f}")
            
            # Metrics chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color='steelblue',
                    text=[f"{v:.3f}" for v in metrics.values()],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Metric",
                yaxis_title="Score",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display visualizations
        st.markdown("---")
        st.subheader("Model Visualizations")
        
        viz_files = list(config.EDA_PLOTS_DIR.glob("*.png"))
        if viz_files:
            cols = st.columns(2)
            for idx, viz_file in enumerate(viz_files):
                with cols[idx % 2]:
                    st.image(str(viz_file), caption=viz_file.stem.replace('_', ' ').title())
        
        # Full report
        st.markdown("---")
        st.subheader("Full Evaluation Report")
        with st.expander("View Complete Report"):
            st.text(eval_report)
    else:
        st.warning("⚠️ No model evaluation found. Train the model first!")
        st.code("python main.py --mode full --quick")

# Tab 4: Make Predictions
with tab4:
    st.header("Make Predictions")
    
    if model is not None:
        st.info("ℹ️ Enter booking details to predict cancellation probability")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
            lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=700, value=100)
            adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
            
        with col2:
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
            babies = st.number_input("Babies", min_value=0, max_value=10, value=0)
            adr = st.number_input("Average Daily Rate ($)", min_value=0.0, max_value=1000.0, value=100.0)
        
        with col3:
            weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
            week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=2)
            special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=0)
        
        if st.button("🎯 Predict Cancellation", type="primary"):
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Note: This is a simplified prediction interface
            # In production, you'd need to preprocess the input the same way as training data
            st.warning("⚠️ Note: Full prediction requires preprocessing the input through the same pipeline used during training. This is a simplified demo.")
            
            # Display input summary
            total_guests = adults + children + babies
            total_nights = weekend_nights + week_nights
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Booking Summary:**
                - Hotel: {hotel}
                - Lead Time: {lead_time} days
                - Total Guests: {total_guests}
                - Total Nights: {total_nights}
                - ADR: ${adr:.2f}
                - Special Requests: {special_requests}
                """)
            
            with col2:
                # Simplified risk assessment based on common patterns
                risk_score = 0
                if lead_time > 200:
                    risk_score += 30
                elif lead_time > 100:
                    risk_score += 15
                
                if adr < 50:
                    risk_score += 20
                elif adr > 200:
                    risk_score += 10
                
                if special_requests == 0:
                    risk_score += 15
                
                if total_nights == 1:
                    risk_score += 10
                
                risk_level = "High" if risk_score > 50 else "Medium" if risk_score > 25 else "Low"
                risk_color = "🔴" if risk_score > 50 else "🟡" if risk_score > 25 else "🟢"
                
                st.markdown(f"""
                **Risk Assessment:**
                - Risk Level: {risk_color} **{risk_level}**
                - Risk Score: {risk_score}/100
                
                **Recommendation:**
                {
                    "Consider offering incentives or flexible policies to reduce cancellation risk." if risk_score > 50
                    else "Monitor booking and send confirmation reminders." if risk_score > 25
                    else "Low risk booking. Standard procedures apply."
                }
                """)
    else:
        st.error("❌ No model loaded. Train the model first!")
        st.code("python main.py --mode full --quick")

# Tab 5: Feature Importance
with tab5:
    st.header("Feature Importance Analysis")
    
    # Check if feature importance plot exists
    importance_plot = config.EDA_PLOTS_DIR / "feature_importance_random_forest.png"
    
    if importance_plot.exists():
        st.image(str(importance_plot), use_container_width=True)
        
        st.markdown("""
        ### Understanding Feature Importance
        
        Feature importance shows which factors most influence booking cancellations:
        
        - **High Importance**: These features have the strongest impact on predictions
        - **Medium Importance**: Moderate influence on cancellation likelihood
        - **Low Importance**: Minimal impact on predictions
        
        **Business Insights:**
        - Focus retention efforts on bookings with high-risk feature combinations
        - Monitor trends in important features over time
        - Use insights to improve booking policies and customer communication
        """)
    else:
        st.warning("⚠️ Feature importance visualization not found. Run the full pipeline to generate it.")
        st.code("python main.py --mode full")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🏨 Hotel Booking Cancellation Prediction Pipeline</p>
    <p>Built with Streamlit | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
