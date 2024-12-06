import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Load preprocessed data, trained model, and preprocessor object
# NOTE: Replace 'X_test.npy', 'y_test.npy', and 'preprocessor.pkl' with specific paths
X_test = np.load('data/X_test.npy')  # Path to preprocessed test features
y_test = np.load('data/y_test.npy')  # Path to preprocessed test labels
preprocessor = joblib.load("models/preprocessor.pkl")  # Path to preprocessor object
model = load_model('models/5g_resource_scheduling_model.h5')  # Path to trained model

# Generate predictions for the test dataset
predictions = model.predict(X_test)

# Decode "Application Behavior" (categorical feature) for visualization
# This step extracts the encoded values and converts them back to human-readable categories
application_behavior_encoded = X_test[:, -len(preprocessor.named_transformers_['cat']['onehot'].categories_[0]):]
application_behavior = np.argmax(application_behavior_encoded, axis=1)  # Decode to categorical index
categories = preprocessor.named_transformers_['cat']['onehot'].categories_[0]  # Original category names

# Sort data for plotting Application Behavior vs Predicted Output
sorted_indices = np.argsort(application_behavior)  # Sort indices by application behavior
sorted_app_behavior = [categories[i] for i in application_behavior[sorted_indices]]
sorted_predictions = predictions[sorted_indices]

# Plot 1: Application Behavior vs Predicted Output
plt.figure(figsize=(10, 6))
plt.plot(sorted_app_behavior, sorted_predictions, marker='o', label="Predicted Output")
plt.title("Application Behavior vs Predicted Output")
plt.xlabel("Application Behavior")
plt.ylabel("Predicted Output")
plt.legend()
plt.grid()
plt.show()

# Extract the indices of numeric features for processing power
# Assuming 'Processing_Power_Client' and 'Processing_Power_Server' are among the numeric features
numeric_features = preprocessor.transformers[0][2]  # List of numeric feature names
client_power_index = numeric_features.index("Processing_Power_Client")
server_power_index = numeric_features.index("Processing_Power_Server")

# Extract scaled values for client and server power from X_test
scaled_client_power = X_test[:, client_power_index].reshape(-1, 1)  # Reshape to 2D for processing
scaled_server_power = X_test[:, server_power_index].reshape(-1, 1)

# Perform inverse transformation on scaled numeric features to get original values
client_power = preprocessor.named_transformers_['num'].inverse_transform(
    np.zeros((scaled_client_power.shape[0], len(numeric_features)))
)[:, client_power_index]

server_power = preprocessor.named_transformers_['num'].inverse_transform(
    np.zeros((scaled_server_power.shape[0], len(numeric_features)))
)[:, server_power_index]

# Sort data for plotting Processing Power (Client)
sorted_indices_client_power = np.argsort(client_power)
sorted_client_power = client_power[sorted_indices_client_power]
sorted_predicted_output_client_power = predictions[sorted_indices_client_power]

# Sort data for plotting Processing Power (Server)
sorted_indices_server_power = np.argsort(server_power)
sorted_server_power = server_power[sorted_indices_server_power]
sorted_predicted_output_server_power = predictions[sorted_indices_server_power]

# Plot 2: Processing Power (Client) vs Predicted Output
plt.figure(figsize=(10, 6))
plt.plot(sorted_client_power, sorted_predicted_output_client_power, marker='o', label="Predicted Output")
plt.title("Processing Power (Client) vs Predicted Output")
plt.xlabel("Processing Power (Client)")
plt.ylabel("Predicted Output")
plt.legend()
plt.grid()
plt.show()

# Plot 3: Processing Power (Server) vs Predicted Output
plt.figure(figsize=(10, 6))
plt.plot(sorted_server_power, sorted_predicted_output_server_power, marker='o', color='purple', label="Predicted Output")
plt.title("Processing Power (Server) vs Predicted Output")
plt.xlabel("Processing Power (Server)")
plt.ylabel("Predicted Output")
plt.legend()
plt.grid()
plt.show()
