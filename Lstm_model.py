import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
file_path = "D:/cyber security/major project/real time/Project/Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path, engine='python')

# Drop non-useful columns
df.drop(columns=['Accident_Index', 'Accident Date', 'Time', 'Datetime'], inplace=True)

# Encode weather conditions
weather_mapping = {
    "Clear": 1, "Sunny": 1,
    "Partly Cloudy": 2, "Cloudy": 3,
    "Fog": 4, "Mist": 5,
    "Rainy": 6, "Snowy": 7, "Stormy": 8
}
df["Weather_Conditions_Num"] = df["Weather_Conditions"].map(weather_mapping).fillna(0)

# Encode target
le = LabelEncoder()
df["Accident_Severity_Num"] = le.fit_transform(df["Accident_Severity"])
joblib.dump(le, 'label_encoder.pkl')

# Feature columns
features = ['Latitude', 'Longitude', 'Number_of_Casualties',
            'Number_of_Vehicles', 'Speed_limit', 'Weather_Conditions_Num']
X = df[features]
y = df["Accident_Severity_Num"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "final_scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# One-hot encode target
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
num_classes = y_train_categorical.shape[1]

# Reshape for LSTM (samples, timesteps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[1])))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_lstm, y_train_categorical, epochs=20, batch_size=32, verbose=1)

# Evaluate
y_pred_proba = model.predict(X_test_lstm)
y_pred = np.argmax(y_pred_proba, axis=1)

# Accuracy and report
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ LSTM Accuracy: {accuracy:.4f}\n")

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save("lstm_model.h5")
print("✅ LSTM model saved as 'lstm_model.h5'")
