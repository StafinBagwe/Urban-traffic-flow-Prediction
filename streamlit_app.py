import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import warnings
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Traffic Flow Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚦 Traffic Flow Prediction System")
st.markdown("### Predict traffic flow using advanced LSTM neural networks")

# Load model and data
@st.cache_resource
def load_models():
    try:
        model = load_model("best_traffic_model.keras")
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        sensor_stats = joblib.load('sensor_stats.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        return model, scaler_X, scaler_y, sensor_stats, feature_cols, None
    except Exception as e:
        return None, None, None, None, None, str(e)

@st.cache_data
def load_dataset():
    try:
        sensors = ['GA0151_A', 'GA0151_C', 'GA0151_D']
        all_data = []
        for sensor in sensors:
            df = pd.read_csv(f"Dataset/{sensor}.csv")
            df['sensor'] = sensor
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str) + ':00:00')
            all_data.append(df)
        full_data = pd.concat(all_data, ignore_index=True)
        return full_data, None
    except Exception as e:
        return None, str(e)

model, scaler_X, scaler_y, sensor_stats, feature_cols, model_error = load_models()
full_data, data_error = load_dataset()

# Check if models loaded successfully
if model is None:
    st.error(f"❌ Error loading model: {model_error}")
    st.info("Please run the training script first to generate model files.")
    st.stop()

if full_data is None:
    st.warning(f"⚠️ Could not load dataset: {data_error}")
else:
    st.success("✅ Model and data loaded successfully!")

# Helper functions
def get_time_period(hour):
    if 22 <= hour or hour <= 5:
        return 0
    elif 6 <= hour <= 9:
        return 1
    elif 10 <= hour <= 16:
        return 2
    elif 17 <= hour <= 19:
        return 3
    else:
        return 4

def get_traffic_multiplier(hour, is_weekend):
    if 22 <= hour or hour <= 5:
        return 0.2 if is_weekend else 0.3
    elif 7 <= hour <= 9:
        return 1.3 if not is_weekend else 0.8
    elif 17 <= hour <= 19:
        return 1.7 if not is_weekend else 1.0
    elif 10 <= hour <= 16:
        return 1.0
    else:
        return 0.9

def create_realistic_sequence(sensor, target_hour, day_of_week, month, full_data):
    sensor_data = full_data[full_data['sensor'] == sensor].copy()
    sensor_data['hour'] = sensor_data['datetime'].dt.hour
    sensor_data['day_of_week'] = sensor_data['datetime'].dt.dayofweek
    sensor_data['month'] = sensor_data['datetime'].dt.month
    
    is_weekend = day_of_week >= 5
    similar_data = sensor_data[
        (sensor_data['day_of_week'] >= 5 if is_weekend else sensor_data['day_of_week'] < 5) &
        (sensor_data['month'] == month)
    ]
    
    if len(similar_data) < 24:
        similar_data = sensor_data
    
    sequence_features = []
    
    for i in range(24):
        seq_hour = (target_hour - 24 + i) % 24
        hour_data = similar_data[similar_data['hour'] == seq_hour]
        
        if len(hour_data) > 0:
            typical_flow = hour_data['flow'].median()
            flow_std = hour_data['flow'].std() if hour_data['flow'].std() > 0 else 1.0
        else:
            typical_flow = similar_data['flow'].median()
            flow_std = similar_data['flow'].std() if similar_data['flow'].std() > 0 else 1.0
        
        if i == 0:
            lag_1 = lag_2 = lag_3 = lag_6 = lag_12 = lag_24 = typical_flow
        else:
            lag_1 = sequence_features[-1]['flow'] if i >= 1 else typical_flow
            lag_2 = sequence_features[-2]['flow'] if i >= 2 else typical_flow
            lag_3 = sequence_features[-3]['flow'] if i >= 3 else typical_flow
            lag_6 = sequence_features[-6]['flow'] if i >= 6 else typical_flow
            lag_12 = sequence_features[-12]['flow'] if i >= 12 else typical_flow
            lag_24 = typical_flow
        
        if i >= 6:
            recent_flows_3 = [sequence_features[j]['flow'] for j in range(i-3, i)]
            recent_flows_6 = [sequence_features[j]['flow'] for j in range(i-6, i)]
            rolling_mean_3 = np.mean(recent_flows_3)
            rolling_std_3 = np.std(recent_flows_3)
            rolling_mean_6 = np.mean(recent_flows_6)
            rolling_max_6 = np.max(recent_flows_6)
            rolling_min_6 = np.min(recent_flows_6)
        else:
            rolling_mean_3 = rolling_mean_6 = typical_flow
            rolling_std_3 = flow_std
            rolling_max_6 = typical_flow * 1.2
            rolling_min_6 = typical_flow * 0.8
        
        hour_sin = np.sin(2 * np.pi * seq_hour / 24)
        hour_cos = np.cos(2 * np.pi * seq_hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        is_weekend_val = 1 if is_weekend else 0
        time_period = get_time_period(seq_hour)
        traffic_multiplier = get_traffic_multiplier(seq_hour, is_weekend)
        
        sensor_GA0151_A = 1 if sensor == 'GA0151_A' else 0
        sensor_GA0151_C = 1 if sensor == 'GA0151_C' else 0
        sensor_GA0151_D = 1 if sensor == 'GA0151_D' else 0
        
        features = {
            'flow': typical_flow,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'is_weekend': is_weekend_val,
            'time_period': time_period,
            'traffic_multiplier': traffic_multiplier,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_3,
            'lag_6': lag_6,
            'lag_12': lag_12,
            'lag_24': lag_24,
            'rolling_mean_3': rolling_mean_3,
            'rolling_std_3': rolling_std_3,
            'rolling_mean_6': rolling_mean_6,
            'rolling_max_6': rolling_max_6,
            'rolling_min_6': rolling_min_6,
            'sensor_GA0151_A': sensor_GA0151_A,
            'sensor_GA0151_C': sensor_GA0151_C,
            'sensor_GA0151_D': sensor_GA0151_D
        }
        
        sequence_features.append(features)
    
    sequence_array = np.array([
        [feat[col] for col in feature_cols]
        for feat in sequence_features
    ])
    
    return sequence_array

def predict_traffic(sensor, hour, day_of_week, month):
    if full_data is not None:
        sequence = create_realistic_sequence(sensor, hour, day_of_week, month, full_data)
    else:
        return None
    
    sequence_scaled = scaler_X.transform(sequence)
    sequence_scaled = sequence_scaled.reshape(1, 24, -1)
    prediction_scaled = model.predict(sequence_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
    return max(0, prediction)

def get_traffic_intensity(flow, sensor, hour, day_of_week):
    stats = sensor_stats[sensor]
    mean = stats['mean']
    
    if 22 <= hour or hour <= 5:
        if flow < 20:
            return "Low", "🟢", "Light night traffic, roads clear"
        elif flow < 30:
            return "Medium", "🟡", "Moderate traffic for night hours"
        else:
            return "High", "🔴", "Unusually heavy traffic for night time"
    
    elif hour == 6:
        if flow < mean * 0.5:
            return "Low", "🟢", "Light early morning traffic"
        elif flow < mean * 0.9:
            return "Medium", "🟡", "Building morning traffic"
        else:
            return "High", "🔴", "Heavy early morning traffic"
    
    elif 7 <= hour <= 9:
        if flow < mean * 0.8:
            return "Low", "🟢", "Light traffic for morning rush"
        elif flow < mean * 1.4:
            return "Medium", "🟡", "Normal morning rush traffic"
        else:
            return "High", "🔴", "Heavy morning rush, expect delays"
    
    elif 10 <= hour <= 16:
        if flow < mean * 0.6:
            return "Low", "🟢", "Light midday traffic"
        elif flow < mean * 1.1:
            return "Medium", "🟡", "Moderate midday traffic"
        else:
            return "High", "🔴", "Heavy midday traffic"
    
    elif 17 <= hour <= 19:
        if flow < mean * 0.9:
            return "Low", "🟢", "Light traffic for evening (unusual)"
        elif flow < mean * 1.3:
            return "Medium", "🟡", "Moderate evening rush"
        else:
            return "High", "🔴", "Heavy evening rush, major congestion expected"
    
    else:
        if flow < mean * 0.5:
            return "Low", "🟢", "Light evening traffic"
        elif flow < mean * 0.9:
            return "Medium", "🟡", "Moderate evening traffic"
        else:
            return "High", "🔴", "Heavy traffic for late evening"

# Sidebar - Input Section
st.sidebar.header("📋 Input Parameters")

# Show dataset range
if full_data is not None:
    min_date = full_data['datetime'].min().date()
    max_date = full_data['datetime'].max().date()
    st.sidebar.info(f"📅 Dataset range:\n{min_date} to {max_date}")

# Sensor selection
sensor = st.sidebar.selectbox(
    "🎯 Select Sensor",
    options=['GA0151_A', 'GA0151_C', 'GA0151_D'],
    help="Choose the traffic sensor location"
)

# Date input
selected_date = st.sidebar.date_input(
    "📅 Select Date",
    value=date.today(),
    min_value=date(2019, 10, 1),
    max_value=date(2030, 12, 31),
    help="Choose the date for prediction"
)

# Hour input
hour = st.sidebar.slider(
    "🕐 Select Hour",
    min_value=0,
    max_value=23,
    value=12,
    help="Choose the hour (0-23)"
)

# Calculate day info
day_of_week = selected_date.weekday()
month = selected_date.month
day_name = selected_date.strftime("%A")

# Display selected info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Selection Summary")
st.sidebar.write(f"**Sensor:** {sensor}")
st.sidebar.write(f"**Date:** {selected_date.strftime('%d %B %Y')}")
st.sidebar.write(f"**Day:** {day_name}")
st.sidebar.write(f"**Time:** {hour}:00")
st.sidebar.write(f"**Month:** {selected_date.strftime('%B')}")

# Predict button
predict_button = st.sidebar.button("🚀 Predict Traffic", use_container_width=True, type="primary")

# Main content area
if predict_button:
    with st.spinner("🔮 Analyzing traffic patterns..."):
        predicted_flow = predict_traffic(sensor, hour, day_of_week, month)
        
        if predicted_flow is not None:
            intensity, emoji, description = get_traffic_intensity(predicted_flow, sensor, hour, day_of_week)
            
            # Results section
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🚗 Predicted Flow",
                    value=f"{predicted_flow:.1f}",
                    delta="vehicles/hour"
                )
            
            with col2:
                intensity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[intensity]
                st.metric(
                    label="🚦 Traffic Intensity",
                    value=f"{intensity_color} {intensity}",
                    delta=None
                )
            
            with col3:
                stats = sensor_stats[sensor]
                st.metric(
                    label="📈 Sensor Average",
                    value=f"{stats['mean']:.1f}",
                    delta="vehicles/hour"
                )
            
            # Description alert
            if intensity == "Low":
                st.success(f"✅ {description}")
            elif intensity == "Medium":
                st.warning(f"⚠️ {description}")
            else:
                st.error(f"🚨 {description}")
            
            # Gauge chart for traffic intensity
            st.markdown("### 🎯 Traffic Intensity Gauge")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_flow,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Vehicles/Hour", 'font': {'size': 24}},
                delta={'reference': stats['mean'], 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, stats['max']]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, stats['mean'] * 0.6], 'color': "lightgreen"},
                        {'range': [stats['mean'] * 0.6, stats['mean'] * 1.3], 'color': "yellow"},
                        {'range': [stats['mean'] * 1.3, stats['max']], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': stats['mean']
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Sensor Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Average Flow', 'Maximum Flow', 'Minimum Flow'],
                    'Value': [f"{stats['mean']:.1f}", f"{stats['max']:.1f}", f"{stats['min']:.1f}"]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### ⏰ Time Context")
                time_periods = {
                    0: "🌙 Night (22-5) - Lowest traffic",
                    1: "🌅 Morning (6-9) - Rush hour",
                    2: "☀️ Midday (10-16) - Normal traffic",
                    3: "🌆 Evening Rush (17-19) - Peak traffic",
                    4: "🌃 Evening (20-21) - Winding down"
                }
                current_period = get_time_period(hour)
                st.info(time_periods[current_period])
                
                if day_of_week >= 5:
                    st.info("📅 Weekend - Traffic typically 30% lower")
                else:
                    st.info("📅 Weekday - Normal traffic patterns")
            
            # Historical comparison (if data available)
            if full_data is not None:
                st.markdown("### 📈 Historical Pattern Comparison")
                
                sensor_data = full_data[full_data['sensor'] == sensor].copy()
                sensor_data['hour'] = sensor_data['datetime'].dt.hour
                hourly_avg = sensor_data.groupby('hour')['flow'].mean().reset_index()
                
                fig2 = px.line(
                    hourly_avg,
                    x='hour',
                    y='flow',
                    title=f'Average Hourly Traffic Pattern for {sensor}',
                    labels={'hour': 'Hour of Day', 'flow': 'Average Flow (vehicles/hour)'}
                )
                
                # Add prediction point
                fig2.add_scatter(
                    x=[hour],
                    y=[predicted_flow],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='Current Prediction'
                )
                
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.error("❌ Prediction failed! Please check if all files are loaded correctly.")

else:
    # Initial state - show instructions
    st.markdown("---")
    st.info("👈 **Select parameters from the sidebar and click 'Predict Traffic' to get started!**")
    
    # Show sample statistics
    if sensor_stats:
        st.markdown("### Sensor Overview:")
        
        sensors_list = ['GA0151_A', 'GA0151_C', 'GA0151_D']
        stats_data = []
        for s in sensors_list:
            stats = sensor_stats[s]
            stats_data.append({
                'Sensor': s,
                'Average Flow': f"{stats['mean']:.1f}",
                'Max Flow': f"{stats['max']:.1f}",
                'Min Flow': f"{stats['min']:.1f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

