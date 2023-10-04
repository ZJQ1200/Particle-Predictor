# Import Statements
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go

# Generate a synthetic dataset with random data
# Please replace these lines with your own dataset if you want
n_samples = 100000  # Number of data points
np.random.seed(0)

data = pd.DataFrame({
    'X': np.random.randint(1, 10, n_samples),
    'Y': np.random.randint(1, 10, n_samples),
    'Z': np.random.randint(1, 10, n_samples),
    'Time': np.arange(0.1, n_samples * 0.1 + 0.1, 0.1)
})

# Create a pattern for Predicted_X, Predicted_Y, and Predicted_Z
# Here, we assume no specific linear relationship
data['Predicted_X'] = np.random.normal(0, 1, n_samples)
data['Predicted_Y'] = np.random.normal(0, 1, n_samples)
data['Predicted_Z'] = np.random.normal(0, 1, n_samples)

# Define the features (X, Y, Z, Time) and the target (Predicted_X, Predicted_Y, Predicted_Z)
X = data[['X', 'Y', 'Z', 'Time']]
y = data[['Predicted_X', 'Predicted_Y', 'Predicted_Z']]

# Standardize Input (X, Y, Z, Time)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Building Neural Network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)  # Output layer with 3 neurons for Predicted_X, Predicted_Y, Predicted_Z
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=0)

# Predict Position Based On Time
def predict_position(input_time):
    # Create a new input with X, Y, Z, and the provided time
    new_input = np.array([[1, 2, 3, input_time]])  # Replace with your input data
    new_input_scaled = scaler.transform(new_input)
    
    # Use the trained model to predict the future position
    predicted_positions = model.predict(new_input_scaled)
    
    return predicted_positions

# Allow For The Manual Input Of Time
input_time = float(input("Enter the time value: "))
predicted_positions = predict_position(input_time)

print("Predicted Positions:")
print(predicted_positions)

#3D Scatter Plot with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=predicted_positions[:, 0],
    y=predicted_positions[:, 1],
    z=predicted_positions[:, 2],
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Predicted Position'
)])

# Set labels for the axes
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Set the title for the plot
fig.update_layout(title='Predicted 3D Position')

# Show the 3D plot for the predicted positions using Plotly
fig.show()
