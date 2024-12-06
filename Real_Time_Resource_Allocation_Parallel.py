import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

# Load the trained model with explicit loss mapping
model_path = "models/5g_resource_scheduling_model.keras"  #generic path for the model
model = load_model(model_path, custom_objects={"mse": MeanSquaredError})

# Load the dataset for statistical properties
dataset_path = "data/Real_time_data.csv"  # generic path for the dataset
original_data = pd.read_csv(dataset_path)

# Separate numeric and categorical columns
numeric_features = [
    "CSI_SNR_dB", "User_Throughput_Historical_Mbps", "User_Throughput_Current_Mbps",
    "QoS_Latency_ms", "QoS_Throughput_Mbps", "QoS_Reliability",
    "Traffic_Pattern_RealTime_Mbps", "Traffic_Pattern_Predicted_Mbps",
    "Resource_Availability", "Data_Rate_Achievable_Mbps", "Latency_Requirements_ms",
    "Base_Station_Load_Percent", "Interference_Levels_dBm", "User_Density_per_km2",
    "Energy_Consumption_J_per_bit", "Mobility_Patterns_kmh", "Processing_Power_Client",
    "Processing_Power_Server"
]
categorical_features = ["Application_Behavior"]

# Extract statistical properties for numeric features
numeric_stats = original_data[numeric_features].describe()

# Extract unique categories for categorical features
categorical_options = {
    col: original_data[col].unique().tolist() for col in categorical_features
}

# Define preprocessing pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessing pipeline on the original data
preprocessor.fit(original_data)

# File to save results
output_file = "output/real_time_predictions_parallel.csv"  # Updated generic output path

# Initialize an empty DataFrame for storing results
columns = numeric_features + categorical_features + ["Predicted_Output"]
results_df = pd.DataFrame(columns=columns)

# Function to handle graceful termination
def signal_handler(sig, frame):
    print("\nExiting program. Saving results...")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Function to generate realistic real-time data
def generate_real_time_data():
    data = {}

    # Generate numeric data based on mean and standard deviation
    for col in numeric_features:
        mean = numeric_stats.loc['mean', col]
        std = numeric_stats.loc['std', col]
        min_val = numeric_stats.loc['min', col]
        max_val = numeric_stats.loc['max', col]

        # Sample within a reasonable range
        data[col] = np.clip(np.random.normal(mean, std), min_val, max_val)

    # Generate categorical data
    for col in categorical_features:
        data[col] = np.random.choice(categorical_options[col])

    return pd.DataFrame([data])

# Function to process a batch of data
def process_batch(batch_size):
    global results_df
    batch_data = [generate_real_time_data() for _ in range(batch_size)]
    batch_df = pd.concat(batch_data, ignore_index=True)

    # Preprocess the batch
    preprocessed_batch = preprocessor.transform(batch_df)

    # Make predictions
    predictions = model.predict(preprocessed_batch)

    # Add predictions to the batch DataFrame
    batch_df["Predicted_Output"] = predictions.flatten()

    # Append to the results DataFrame
    results_df = pd.concat([results_df, batch_df], ignore_index=True)

    # Save updated results to the CSV file
    results_df.to_csv(output_file, index=False)

    return batch_df

# Simulate real-time data processing with parallel execution
def real_time_prediction_loop(batch_size=10, num_workers=2):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while True:
            future = executor.submit(process_batch, batch_size)
            batch_result = future.result()  # Block until the batch is processed

            # Output the batch result to the console
            print(f"Processed Batch:\n{batch_result}")

            # Simulate a delay for real-time generation
            time.sleep(0.1)  # Adjust as needed for batch frequency

# Run the real-time prediction loop
if __name__ == "__main__":
    print("Starting real-time prediction with realistic input generation...")
    print("Press Ctrl+C to exit and save results.")
    real_time_prediction_loop(batch_size=10, num_workers=4)
