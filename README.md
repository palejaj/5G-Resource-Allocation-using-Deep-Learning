# 5G Resource Allocation Using Deep Learning

## Project Overview
This project aims to **optimally schedule resources in a 5G network** using an efficient **Feedforward Neural Network (FFN)**. The unique feature of this model is its ability to adapt to the dynamic and non-sequential nature of real-life 5G network data, where input characteristics and requirements change over time.

---

## Key Features
- **Input Attributes**: Considers 15 input attributes for decision-making, ensuring accurate and efficient resource allocation.
- **Real-Time Capability**: Allocates resources dynamically in real-time, making it ideal for practical deployment scenarios.

---

## Project Components
1. **Model Trainer (`model_trainer.py`)**:
   - Preprocesses the dataset (`5g_Resource_Allocation_Dataset.csv`).
   - Splits the data into training and testing sets.
   - Saves preprocessed test data (`X-test.npy`, `y-test.npy`) for analysis.
   - Trains the Feedforward Neural Network (FFN) on the data.
   - Saves the trained model as `5g_resource_scheduling_model.h5` for inference.

2. **Plot Script (`plot.py`)**:
   - Uses preprocessed test data (`X-test.npy`, `y-test.npy`) to generate visualizations.
   - Produces insights through plots and analysis for further evaluation.

3. **Real-Time Resource Allocation (`Real_Time_Resource_Allocation_Parallel.py`)**:
   - Loads the trained model to make real-time predictions.
   - Processes input data from `Real_time_data.csv`.
   - Outputs the results into a CSV file named `real_time_predictions_parallel.csv`, pairing input attributes with predicted outputs.

---

## Setup Instructions
### Environment Setup
1. Install Python 3.9 or later.
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Files
- Ensure `5g_Resource_Allocation_Dataset.csv` and `Real_time_data.csv` are placed in the `/data` folder.

---

## Usage Instructions
### 1. Train the Model
Run the `model_trainer.py` script to preprocess the data and train the FFN:
```bash
python src/model_trainer.py
```
Outputs:
- Trained model: `5g_resource_scheduling_model.h5`.
- Preprocessed test data: `X-test.npy`, `y-test.npy`.

### 2. Generate Visualizations
Run `plot.py` to analyze the test data and generate plots:
```bash
python src/plot.py
```

### 3. Make Real-Time Predictions
Run `Real_Time_Resource_Allocation_Parallel.py` to allocate resources dynamically using real-time input data:
```bash
python src/Real_Time_Resource_Allocation_Parallel.py
```
Outputs:
- Results saved to `real_time_predictions_parallel.csv`.

---

## Output Description
1. **Preprocessed Data**:
   - `X-test.npy`: Test input features.
   - `y-test.npy`: Test output labels.
2. **Trained Model**:
   - `5g_resource_scheduling_model.h5`: Neural network model for inference.
3. **Prediction Results**:
   - `real_time_predictions_parallel.csv`: Contains input attributes and corresponding predicted outputs.

---

## Visualizations
The `plot.py` script generates detailed visualizations for insights into the dataset and the model's performance. Key plots include:
- Attribute distributions.
- Performance metrics of the model.

---

## Dataset
- **`5g_Resource_Allocation_Dataset.csv`**:
   - Contains 15 attributes related to resource allocation in a 5G network.
- **`Real_time_data.csv`**:
   - Includes real-time input data for testing predictions.

---

## Contributions
This project is open for contributions. Feel free to fork the repository and submit pull requests.

---

## Future Work
- Expand the model to support **multiple base stations** to handle diverse and distributed network environments.
- Train the model using additional attributes such as **geographical location** and other environmental factors that may influence resource allocation efficiency.
- Explore multi-task learning to support different network scenarios.
- Integrate reinforcement learning to dynamically improve resource allocation decisions.
- Test and refine the model on larger, more diverse datasets for broader applicability.

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

---


