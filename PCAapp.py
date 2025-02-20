import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title and Description
st.title('Bill Authenticity Prediction')
st.write('Enter the features of the bill to predict whether it is Genuine or Fake.')

# Sample Data for Training
data = pd.read_csv("bills.csv")
df = pd.DataFrame(data)

# Features and Target
X = df.drop(['is_genuine'], axis=1)
y = df['is_genuine']

# Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# New Data Input Section
st.write('## Input New Data for Prediction')
input_data = []

# Asking for each feature value
diagonal = st.number_input('Diagonal', value=171.0)
input_data.append(diagonal)

height_left = st.number_input('Height Left', value=104.0)
input_data.append(height_left)

height_right = st.number_input('Height Right', value=104.0)
input_data.append(height_right)

margin_low = st.number_input('Margin Low', value=4.0)
input_data.append(margin_low)

margin_up = st.number_input('Margin Up', value=3.0)
input_data.append(margin_up)

length = st.number_input('Length', value=113.0)
input_data.append(length)

# Prediction Button
if st.button('Predict'):
    # Standardize Input Data
    new_data_scaled = scaler.transform([input_data])
    
    # Make Prediction
    new_pred = model.predict(new_data_scaled)
    prediction_label = 'Genuine' if new_pred[0] == 1 else 'Fake'
    st.write(f'### Prediction: {prediction_label}')
