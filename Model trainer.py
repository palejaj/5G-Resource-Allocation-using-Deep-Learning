import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# Load the dataset
file_path = "data/5g_Resource_Allocation_Dataset.csv"  # Generic file path for the dataset
data = pd.read_csv(file_path)

# Separate features and target
X = data.iloc[:, :-1]  # All columns except the last
y = data.iloc[:, -1]   # The last column (target variable)

# Separate numeric and categorical features
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = ["Application_Behavior"]  # Assuming Application_Behavior is the only categorical column

# Define preprocessing pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Standardize numeric features
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save preprocessing object for later use
joblib.dump(preprocessor, "models/preprocessor.pkl")  # Save preprocessor in the models folder

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dropout(0.4),  # Increased dropout to reduce overfitting
    Dense(64, activation='relu'),  # Hidden layer 1
    Dropout(0.4),  # Dropout layer
    Dense(32, activation='relu'),  # Hidden layer 2
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE for loss, MAE as metric

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop training early
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)  # Reduce LR on plateau

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the model and processed data
model.save('models/5g_resource_scheduling_model.keras')  # Save model in the models folder
np.save('data/X_test.npy', X_test)  # Save preprocessed test features
np.save('data/y_test.npy', y_test)  # Save preprocessed test labels
print("Model training complete and saved as '5g_resource_scheduling_model.keras'.")
