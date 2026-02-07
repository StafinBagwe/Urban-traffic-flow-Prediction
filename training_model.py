import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Improved Traffic Flow Model Training...")
print("=" * 60)

# Load datasets
sensors = ['GA0151_A', 'GA0151_C', 'GA0151_D']
all_data = []

for sensor in sensors:
    df = pd.read_csv(f"Dataset/{sensor}.csv")
    df['sensor'] = sensor
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str) + ':00:00')
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data = data.sort_values(['sensor', 'datetime']).reset_index(drop=True)

print(f"✅ Loaded {len(data)} records from {len(sensors)} sensors")
print(f"📅 Date range: {data['datetime'].min()} to {data['datetime'].max()}")

# Remove outliers using IQR method
def remove_outliers(df, column='flow'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_len = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    removed = original_len - len(df_clean)
    
    print(f"🧹 Removed {removed} outliers ({removed/original_len*100:.2f}%)")
    return df_clean

data = remove_outliers(data)

# Extract time features
data['hour'] = data['datetime'].dt.hour
#data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month
data['day_of_month'] = data['datetime'].dt.day
#data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# CRITICAL: Add time period categories
def get_time_period(hour):
    if 22 <= hour or hour <= 5:
        return 0  # Night - LOW traffic expected
    elif 6 <= hour <= 9:
        return 1  # Morning transition/rush
    elif 10 <= hour <= 16:
        return 2  # Midday
    elif 17 <= hour <= 19:
        return 3  # Evening rush - HIGH traffic expected
    else:  # 20-21
        return 4  # Evening

data['time_period'] = data['hour'].apply(get_time_period)

# Add time period multipliers (expected traffic level)
def get_traffic_multiplier(hour, is_weekend):
    if 22 <= hour or hour <= 5:  # Night
        return 0.2 if is_weekend else 0.3
    elif 7 <= hour <= 9:  # Morning rush
        return 1.3 if not is_weekend else 0.8
    elif 17 <= hour <= 19:  # Evening rush
        return 1.7 if not is_weekend else 1.0
    elif 10 <= hour <= 16:  # Midday
        return 1.0
    else:
        return 0.9

data['traffic_multiplier'] = data.apply(
    lambda row: get_traffic_multiplier(row['hour'], row['is_weekend']), axis=1
)

# Cyclical encoding
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# One-hot encode sensors
sensor_dummies = pd.get_dummies(data['sensor'], prefix='sensor')
data = pd.concat([data, sensor_dummies], axis=1)

# Create lag features per sensor
print("⏳ Creating lag features...")
for sensor in sensors:
    sensor_mask = data['sensor'] == sensor
    data.loc[sensor_mask, 'lag_1'] = data.loc[sensor_mask, 'flow'].shift(1)
    data.loc[sensor_mask, 'lag_2'] = data.loc[sensor_mask, 'flow'].shift(2)
    data.loc[sensor_mask, 'lag_3'] = data.loc[sensor_mask, 'flow'].shift(3)
    data.loc[sensor_mask, 'lag_6'] = data.loc[sensor_mask, 'flow'].shift(6)
    data.loc[sensor_mask, 'lag_12'] = data.loc[sensor_mask, 'flow'].shift(12)
    data.loc[sensor_mask, 'lag_24'] = data.loc[sensor_mask, 'flow'].shift(24)

# Rolling statistics per sensor
print("⏳ Creating rolling statistics...")
for sensor in sensors:
    sensor_mask = data['sensor'] == sensor
    data.loc[sensor_mask, 'rolling_mean_3'] = data.loc[sensor_mask, 'flow'].shift(1).rolling(3).mean()
    data.loc[sensor_mask, 'rolling_std_3'] = data.loc[sensor_mask, 'flow'].shift(1).rolling(3).std()
    data.loc[sensor_mask, 'rolling_mean_6'] = data.loc[sensor_mask, 'flow'].shift(1).rolling(6).mean()
    data.loc[sensor_mask, 'rolling_max_6'] = data.loc[sensor_mask, 'flow'].shift(1).rolling(6).max()
    data.loc[sensor_mask, 'rolling_min_6'] = data.loc[sensor_mask, 'flow'].shift(1).rolling(6).min()

# Fill NaN values
data = data.fillna(method='bfill').fillna(method='ffill').fillna(0)

print("✅ Feature engineering completed")

# Prepare features - INCLUDING time_period and traffic_multiplier
feature_cols = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'is_weekend', 'time_period', 'traffic_multiplier',
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
    'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_max_6', 'rolling_min_6',
    'sensor_GA0151_A', 'sensor_GA0151_C', 'sensor_GA0151_D'
]

X = data[feature_cols].values
y = data['flow'].values

# Use RobustScaler (better for outliers)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Create sequences for LSTM
def create_sequences(X, y, seq_length=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

print(f"✅ Created sequences: {X_seq.shape}")

# Train-test split (80-20, time-aware split)
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Build improved model
print("\n🏗️  Building neural network...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, len(feature_cols))),
    Dropout(0.3),
    BatchNormalization(),
    
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

# Custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='huber',  # More robust to outliers than MSE
    metrics=['mae', 'mse']
)

print("\n📊 Model Architecture:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_traffic_model.keras',
    save_best_only=True,
    monitor='val_loss',
    verbose=1,
    save_weights_only=False
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# Train
print("\n🏋️  Training model...")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Load best model
model = tf.keras.models.load_model('best_traffic_model.keras')

# Evaluate on test set
print("\n" + "=" * 60)
print("📈 MODEL EVALUATION ON TEST SET")
print("=" * 60)

# Predictions on scaled data
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to real values
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler_y.inverse_transform(y_pred_scaled).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)

# Calculate accuracy percentage
mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-10))) * 100
accuracy = 100 - mape

print(f"\n📊 Real-World Performance Metrics:")
print(f"   • Mean Absolute Error (MAE): {mae:.2f} vehicles/hour")
print(f"   • Root Mean Squared Error (RMSE): {rmse:.2f} vehicles/hour")
print(f"   • R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"   • Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"   • Model Accuracy: {accuracy:.2f}%")

# Breakdown by time period
print(f"\n📊 Accuracy by Time Period:")
# Get time periods from test data
test_start_idx = split_idx + SEQ_LENGTH
time_periods_test = data.iloc[test_start_idx:test_start_idx+len(y_test)]['time_period'].values
time_period_names = {0: 'Night (22-5)', 1: 'Morning (6-9)', 2: 'Midday (10-16)', 
                     3: 'Evening Rush (17-19)', 4: 'Evening (20-21)'}

for period in sorted(time_period_names.keys()):
    mask = time_periods_test == period
    if mask.sum() > 0:
        period_mae = mean_absolute_error(y_test_real[mask], y_pred_real[mask])
        period_r2 = r2_score(y_test_real[mask], y_pred_real[mask])
        print(f"   • {time_period_names[period]:20s}: MAE={period_mae:.2f}, R²={period_r2:.3f}")

# Show prediction examples
print(f"\n📊 Sample Predictions (First 10 from Test Set):")
print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
print("-" * 45)
for i in range(min(10, len(y_test_real))):
    actual = y_test_real[i]
    pred = y_pred_real[i]
    error = abs(actual - pred)
    error_pct = (error / (actual + 1e-10)) * 100
    print(f"{actual:<10.1f} {pred:<10.1f} {error:<10.1f} {error_pct:<10.1f}%")

# Save everything
print(f"\n💾 Saving model and preprocessing objects...")
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Save sensor statistics
sensor_stats = {}
for sensor in sensors:
    sensor_df = data[data['sensor'] == sensor]
    sensor_stats[sensor] = {
        'mean': sensor_df['flow'].mean(),
        'std': sensor_df['flow'].std(),
        'max': sensor_df['flow'].max(),
        'min': sensor_df['flow'].min(),
        'last_24': sensor_df.tail(24)['flow'].values.tolist()
    }

joblib.dump(sensor_stats, 'sensor_stats.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"✅ Model saved: best_traffic_model.keras")
print(f"✅ Scalers saved: scaler_X.pkl, scaler_y.pkl")
print(f"✅ Metadata saved: sensor_stats.pkl, feature_cols.pkl")
print(f"\n🎯 Final Model Accuracy: {accuracy:.2f}%")
print(f"🎯 Average Prediction Error: ±{mae:.1f} vehicles/hour")
print(f"\n🚀 Ready to use prediction script!")
print("=" * 60)