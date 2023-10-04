# Import Statements
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go

# Generate a synthetic dataset with random data
# Please replace these lines with your own dataset if you want
n_samples = 1000  # Number of data points
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

# Function to predict position based on time
def predict_position(input_time):
    # Create a new input with X, Y, Z, and the provided time
    new_input = np.array([[1, 2, 3, input_time]])  # Replace with your input data
    new_input_scaled = scaler.transform(new_input)
    
    # Use the trained model to predict the future position
    predicted_positions = model.predict(new_input_scaled)
    
    return predicted_positions

# Allow for manual input of time and period
input_time = float(input("Enter the starting time value: "))
period = float(input("Enter the period of time: "))
end_time = input_time + period

# Create a list to store predicted positions and time points over time
time_points = []
predicted_positions_list = []

while input_time <= end_time:
    predicted_positions = predict_position(input_time)
    time_points.append(input_time)
    predicted_positions_list.append(predicted_positions)
    input_time += 0.1  # You can adjust the time step as needed

print("Predicted Positions Over Time:")
for t, positions in zip(time_points, predicted_positions_list):
    print(f"Time: {t:.2f}, Positions: {positions}")

# Create a 3D scatter plot with Plotly to visualize the positions over time
fig = go.Figure()

# Create an empty trace for the particle
particle_trace = go.Scatter3d(
    x=[],
    y=[],
    z=[],
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Particle'
)

fig.add_trace(particle_trace)

# Set labels for the axes
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Set the title for the plot
fig.update_layout(title='Predicted 3D Position Over Time')

# Define animation frames
frames = []

for i in range(len(time_points)):
    frame = go.Frame(
        data=[go.Scatter3d(
            x=[predicted_positions_list[i][0][0]],
            y=[predicted_positions_list[i][0][1]],
            z=[predicted_positions_list[i][0][2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            name=f'Time: {time_points[i]:.2f}'
        )],
        name=f'Time: {time_points[i]:.2f}'
    )
    frames.append(frame)

# Update frames
fig.update(frames=frames)

# Set animation duration and show the animated 3D plot for the positions over time using Plotly
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')])])])

fig.show()
