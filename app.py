"""
AI Startup Failure Early Warning System
A production-ready Streamlit web app for predicting startup failure risk using Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="AI Startup Failure Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        font-weight: bold;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .status-safe {
        color: #27ae60;
        font-weight: bold;
        font-size: 24px;
    }
    .status-warning {
        color: #f39c12;
        font-weight: bold;
        font-size: 24px;
    }
    .status-critical {
        color: #e74c3c;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def generate_synthetic_dataset():
    """Generate synthetic startup dataset for training the ML model"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    revenue = np.random.exponential(500000, n_samples)
    expenses = revenue * np.random.uniform(0.5, 1.5, n_samples)
    growth_rate = np.random.normal(15, 20, n_samples)
    burn_rate = np.random.uniform(10, 80, n_samples)
    employees = np.random.randint(5, 500, n_samples)
    customer_growth = np.random.normal(10, 15, n_samples)
    market_trend = np.random.uniform(0, 100, n_samples)
    
    # Create target variable based on business logic
    failure_risk = np.zeros(n_samples)
    for i in range(n_samples):
        risk_score = 0
        # High expenses vs revenue
        if expenses[i] > revenue[i] * 1.2:
            risk_score += 30
        # Negative growth
        if growth_rate[i] < 0:
            risk_score += 25
        # High burn rate
        if burn_rate[i] > 60:
            risk_score += 20
        # Low customer growth
        if customer_growth[i] < 5:
            risk_score += 15
        # Poor market trend
        if market_trend[i] < 40:
            risk_score += 10
            
        failure_risk[i] = 1 if risk_score > 50 else 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'revenue': revenue,
        'expenses': expenses,
        'growth_rate': growth_rate,
        'burn_rate': burn_rate,
        'employees': employees,
        'customer_growth': customer_growth,
        'market_trend': market_trend,
        'failure_risk': failure_risk
    })
    
    return df


@st.cache_resource
def train_model():
    """Train RandomForestClassifier on synthetic data"""
    # Generate dataset
    df = generate_synthetic_dataset()
    
    # Prepare features and target
    X = df.drop('failure_risk', axis=1)
    y = df['failure_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    
    return model, X.columns.tolist(), accuracy


def predict_failure_risk(model, input_data):
    """Predict startup failure risk"""
    # Get prediction probability
    prediction_proba = model.predict_proba(input_data)[0]
    failure_probability = prediction_proba[1] * 100
    survival_probability = prediction_proba[0] * 100
    
    # Calculate health score (inverse of failure risk)
    health_score = 100 - failure_probability
    
    # Determine status
    if failure_probability < 40:
        status = "Safe"
        status_color = "status-safe"
        status_emoji = "✅"
    elif failure_probability < 70:
        status = "Warning"
        status_color = "status-warning"
        status_emoji = "⚠️"
    else:
        status = "Critical"
        status_color = "status-critical"
        status_emoji = "🚨"
    
    return {
        'failure_risk': failure_probability,
        'survival_probability': survival_probability,
        'health_score': health_score,
        'status': status,
        'status_color': status_color,
        'status_emoji': status_emoji
    }


def get_recommendations(failure_risk):
    """Generate AI recommendations based on risk level"""
    if failure_risk < 40:
        return [
            "✅ Startup is healthy. Continue current growth strategy.",
            "💡 Maintain positive cash flow and sustainable growth.",
            "📈 Focus on scaling operations and market expansion.",
            "🎯 Continue monitoring key performance indicators."
        ]
    elif failure_risk < 70:
        return [
            "⚠️ Monitor expenses and improve customer acquisition.",
            "💰 Review and optimize operational costs.",
            "📊 Increase revenue streams and diversify income sources.",
            "👥 Focus on customer retention and satisfaction.",
            "🔍 Conduct market analysis to identify growth opportunities."
        ]
    else:
        return [
            "🚨 High failure risk. Reduce burn rate immediately.",
            "⚡ Critical: Cut non-essential expenses NOW.",
            "💼 Consider strategic partnerships or funding options.",
            "📉 Implement emergency cost-reduction measures.",
            "🔄 Pivot business model if current strategy isn't working.",
            "👨‍💼 Seek advisory support from experienced mentors."
        ]


def create_gauge_chart(value, title):
    """Create a gauge chart using Plotly"""
    # Determine color based on value
    if value < 40:
        color = "green"
    elif value < 70:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_risk_bar_chart(predictions):
    """Create risk comparison bar chart"""
    metrics = ['Health Score', 'Survival Probability', 'Failure Risk']
    values = [
        predictions['health_score'],
        predictions['survival_probability'],
        predictions['failure_risk']
    ]
    colors = ['green', 'blue', 'red']
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Startup Metrics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f5f7fa"
    )
    
    return fig


def create_feature_importance_chart(model, feature_names):
    """Create feature importance chart"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[feature_names[i] for i in indices],
            y=[importances[i] for i in indices],
            marker_color='lightblue',
            text=[f"{importances[i]:.3f}" for i in indices],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Feature Importance in Risk Prediction",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f5f7fa"
    )
    
    return fig


def create_risk_distribution_chart():
    """Create risk distribution chart using training data"""
    df = generate_synthetic_dataset()
    
    fig = px.histogram(
        df,
        x='failure_risk',
        color='failure_risk',
        labels={'failure_risk': 'Risk Category'},
        title='Risk Distribution in Training Data',
        color_discrete_map={0: 'green', 1: 'red'},
        category_orders={'failure_risk': [0, 1]}
    )
    
    fig.update_layout(
        xaxis_title="Risk Category (0=Safe, 1=High Risk)",
        yaxis_title="Number of Startups",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f5f7fa",
        showlegend=False
    )
    
    return fig


def dashboard_page(model, feature_names, accuracy):
    """Main Dashboard Page"""
    st.title("🏢 AI Startup Failure Early Warning System")
    st.markdown("### Real-time Risk Assessment Dashboard")
    
    # Display model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🤖 Model: RandomForestClassifier")
    with col2:
        st.info(f"📊 Training Accuracy: {accuracy*100:.2f}%")
    with col3:
        st.info(f"📈 Training Samples: 1000")
    
    st.markdown("---")
    
    # Sidebar inputs
    st.sidebar.header("📊 Startup Metrics Input")
    st.sidebar.markdown("Enter your startup's current metrics:")
    
    revenue = st.sidebar.number_input(
        "Monthly Revenue (₹)",
        min_value=0,
        max_value=100000000,
        value=500000,
        step=10000,
        help="Total monthly revenue in rupees"
    )
    
    expenses = st.sidebar.number_input(
        "Monthly Expenses (₹)",
        min_value=0,
        max_value=100000000,
        value=400000,
        step=10000,
        help="Total monthly expenses in rupees"
    )
    
    growth_rate = st.sidebar.slider(
        "Growth Rate (%)",
        min_value=-50.0,
        max_value=200.0,
        value=15.0,
        step=0.5,
        help="Monthly growth rate percentage"
    )
    
    burn_rate = st.sidebar.slider(
        "Burn Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Cash burn rate percentage"
    )
    
    employees = st.sidebar.number_input(
        "Employee Count",
        min_value=1,
        max_value=10000,
        value=50,
        step=1,
        help="Total number of employees"
    )
    
    customer_growth = st.sidebar.slider(
        "Customer Growth (%)",
        min_value=-50.0,
        max_value=200.0,
        value=10.0,
        step=0.5,
        help="Monthly customer growth percentage"
    )
    
    market_trend = st.sidebar.slider(
        "Market Trend Score (0-100)",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        help="Market trend score from 0 to 100"
    )
    
    # Analyze button
    analyze_button = st.sidebar.button("🔍 Analyze Startup Risk", type="primary", use_container_width=True)
    
    if analyze_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'revenue': [revenue],
            'expenses': [expenses],
            'growth_rate': [growth_rate],
            'burn_rate': [burn_rate],
            'employees': [employees],
            'customer_growth': [customer_growth],
            'market_trend': [market_trend]
        })
        
        # Get predictions
        predictions = predict_failure_risk(model, input_data)
        
        # Store in session state
        st.session_state['predictions'] = predictions
        st.session_state['input_data'] = input_data
    
    # Display results if available
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        
        # Status banner
        st.markdown(f"""
        <div style='background-color: {"#d4edda" if predictions["failure_risk"] < 40 else "#fff3cd" if predictions["failure_risk"] < 70 else "#f8d7da"}; 
                    padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
            <span class='{predictions["status_color"]}'>{predictions['status_emoji']} Status: {predictions['status']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Executive Dashboard Metrics
        st.markdown("### 📊 Executive Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💚 Startup Health Score",
                value=f"{predictions['health_score']:.1f}%",
                delta=f"{predictions['health_score'] - 50:.1f}% vs baseline"
            )
        
        with col2:
            st.metric(
                label="🚨 Failure Risk Score",
                value=f"{predictions['failure_risk']:.1f}%",
                delta=f"{predictions['failure_risk'] - 50:.1f}% vs baseline",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="🛡️ Survival Probability",
                value=f"{predictions['survival_probability']:.1f}%",
                delta=f"{predictions['survival_probability'] - 50:.1f}% vs baseline"
            )
        
        with col4:
            st.metric(
                label="📍 Status Indicator",
                value=predictions['status'],
                delta=predictions['status_emoji']
            )
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Risk Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_gauge_chart(predictions['failure_risk'], "Failure Risk Score"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_gauge_chart(predictions['health_score'], "Startup Health Score"),
                use_container_width=True
            )
        
        # Risk bar chart
        st.plotly_chart(create_risk_bar_chart(predictions), use_container_width=True)
        
        st.markdown("---")
        
        # AI Recommendations
        st.markdown("### 🤖 AI-Powered Recommendations")
        recommendations = get_recommendations(predictions['failure_risk'])
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
    else:
        # Show placeholder when no analysis yet
        st.info("👈 Enter your startup metrics in the sidebar and click 'Analyze Startup Risk' to get started!")
        
        # Show demo info
        st.markdown("### 🎯 What This System Does")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📊 Risk Analysis
            - Predicts failure probability
            - Calculates health scores
            - Identifies critical metrics
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 Real-time Insights
            - Instant risk assessment
            - Visual dashboards
            - Status indicators
            """)
        
        with col3:
            st.markdown("""
            #### 🤖 AI Recommendations
            - Actionable advice
            - Risk mitigation strategies
            - Growth optimization tips
            """)


def analytics_page(model, feature_names):
    """Analytics Page"""
    st.title("📊 Analytics & Insights")
    st.markdown("### Deep Dive into Model Performance and Data Patterns")
    
    # Feature Importance
    st.markdown("#### 🎯 Feature Importance Analysis")
    st.markdown("Understanding which factors most influence startup failure risk:")
    st.plotly_chart(
        create_feature_importance_chart(model, feature_names),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Risk Distribution
    st.markdown("#### 📈 Risk Distribution in Training Data")
    st.markdown("Distribution of safe vs. high-risk startups in our training dataset:")
    st.plotly_chart(
        create_risk_distribution_chart(),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Training Data Statistics
    st.markdown("#### 📋 Training Data Statistics")
    df = generate_synthetic_dataset()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Overview:**")
        st.write(f"- Total samples: {len(df)}")
        st.write(f"- Safe startups: {len(df[df['failure_risk'] == 0])} ({len(df[df['failure_risk'] == 0])/len(df)*100:.1f}%)")
        st.write(f"- High-risk startups: {len(df[df['failure_risk'] == 1])} ({len(df[df['failure_risk'] == 1])/len(df)*100:.1f}%)")
    
    with col2:
        st.markdown("**Feature Statistics:**")
        st.dataframe(df.describe().round(2), use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.markdown("#### 🔥 Feature Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = df.drop('failure_risk', axis=1).corr()
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=500,
        margin=dict(l=100, r=40, t=60, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application function"""
    # Train model (cached)
    model, feature_names, accuracy = train_model()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Analytics"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Route to appropriate page
    if page == "Dashboard":
        dashboard_page(model, feature_names, accuracy)
    elif page == "Analytics":
        analytics_page(model, feature_names)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 💡 About
    This AI system uses **Machine Learning** to predict startup failure risk in real-time.
    
    **Powered by:**
    - RandomForestClassifier
    - Python & Streamlit
    - Plotly & Matplotlib
    
    **Version:** 1.0.0
    """)


if __name__ == "__main__":
    main()
