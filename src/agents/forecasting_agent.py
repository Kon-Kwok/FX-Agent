# FX-Agent/src/agents/forecasting_agent.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

# Project-specific imports
from src.utils.data_processor import DataProcessor
from src.utils.hyperparameter_optimizer import HyperparameterOptimizer
from src.models.model_factory import ModelFactory

def forecasting_agent(state: Dict) -> Dict:
    """
    Executes a complete workflow integrating data processing, hyperparameter optimization,
    and model forecasting.

    Args:
        state (Dict): The current state of the workflow, which must include:
            - 'data': {
                'structured_data': pd.DataFrame,
                'selected_features': List[str],
                'target_column': str
            }
            - 'plan': {'model_name': str}

    Returns:
        Dict: The updated state, including 'best_hyperparameters' and 'forecast_result'.
    """
    print("--- Running Forecasting Agent ---")

    # 1. Retrieve data and plan from the state
    structured_data = state.get("data", {}).get("structured_data")
    selected_features = state.get("data", {}).get("selected_features")
    target_column = state.get("data", {}).get("target_column")
    model_name = state.get("plan", {}).get("model_name")

    if structured_data is None or not selected_features or not target_column or not model_name:
        raise ValueError("State is missing required data for forecasting: structured_data, selected_features, target_column, or model_name.")

    # 2. Data Processing
    print("--- Step 1: Processing Data ---")
    data_processor = DataProcessor(sequence_length=60, test_size=0.2)
    processed_data = data_processor.process_data(
        df=structured_data,
        target_column=target_column,
        feature_columns=selected_features
    )
    X_train, y_train = processed_data['X_train'], processed_data['y_train']
    X_test, y_test = processed_data['X_test'], processed_data['y_test']
    scaler = processed_data['scaler']
    
    # Reshape y_train and y_test to the correct shape (n_samples, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    print(f"Data processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Hyperparameter Optimization
    print(f"--- Step 2: Optimizing Hyperparameters for {model_name} ---")
    optimizer = HyperparameterOptimizer(X_train, y_train, X_test, y_test)
    # Note: n_trials is set to a small value for demonstration purposes.
    best_params = optimizer.optimize(model_name=model_name, n_trials=25)
    state['best_hyperparameters'] = best_params
    print(f"Best hyperparameters found: {best_params}")

    # 4. Final Model Training
    print("--- Step 3: Training Final Model with Best Hyperparameters ---")
    
    # Prepare parameters for model creation
    final_model_params = best_params.copy()
    final_model_params['input_dim'] = X_train.shape[2]
    final_model_params['output_dim'] = y_train.shape[1]

    final_model = ModelFactory.create_model(model_name, **final_model_params)
    
    criterion = nn.MSELoss()
    model_optimizer = optim.Adam(final_model.parameters(), lr=final_model_params.get('learning_rate', 0.001))
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    # Standard PyTorch training loop
    num_epochs = 100  # Use more epochs for final training
    batch_size = 64
    for epoch in range(num_epochs):
        final_model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            model_optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            model_optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    print("Final model training complete.")

    # 5. Generate Forecasts
    print("--- Step 4: Generating Forecasts ---")
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float()
        predictions_scaled = final_model(X_test_tensor).numpy()

    # 6. Inverse transform the predictions
    # Create a dummy array with the same shape as the scaler expects
    # The scaler expects an input of shape (n_samples, n_features)
    # We need to place our predictions in the column corresponding to the target
    num_features = len(selected_features) + 1 # features + target
    dummy_features = np.zeros((predictions_scaled.shape[0], num_features))
    
    # Assume the target column was the last one during scaling
    dummy_features[:, -1] = predictions_scaled.flatten()
    
    # Perform inverse transformation
    predictions_inversed = scaler.inverse_transform(dummy_features)
    
    # Extract the inverse-transformed predictions
    final_predictions = predictions_inversed[:, -1]

    state['forecast_result'] = final_predictions.tolist()
    print(f"Forecast generated and saved to state. Sample: {final_predictions[:5]}...")
    
    print("--- Forecasting Agent Finished ---")
    return state

if __name__ == '__main__':
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Please run: pip install yfinance optuna")
        exit()

    # 1. Create a mock state
    print("--- Preparing Demo State ---")
    # Download data
    raw_data = yf.download('EURUSD=X', start='2022-01-01', end='2023-12-31', interval='1d')
    
    if raw_data.empty:
        print("Failed to download data, creating dummy data.")
        date_rng = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        raw_data = pd.DataFrame(date_rng, columns=['Date'])
        raw_data['Open'] = np.random.uniform(1.0, 1.2, size=(len(date_rng)))
        raw_data['High'] = raw_data['Open'] + np.random.uniform(0, 0.05, size=(len(date_rng)))
        raw_data['Low'] = raw_data['Open'] - np.random.uniform(0, 0.05, size=(len(date_rng)))
        raw_data['Close'] = np.random.uniform(raw_data['Low'], raw_data['High'], size=(len(date_rng)))
        raw_data['Volume'] = np.random.randint(10000, 100000, size=(len(date_rng)))
        raw_data.set_index('Date', inplace=True)

    # Define the state
    demo_state = {
        "data": {
            "structured_data": raw_data,
            "selected_features": ['Open', 'High', 'Low', 'Volume'],
            "target_column": "Close"
        },
        "plan": {
            "model_name": "LSTM" # Can be changed to "Transformer" for testing
        },
        "best_hyperparameters": None,
        "forecast_result": None
    }
    print("Demo state created.")

    # 2. Call the forecasting_agent
    final_state = forecasting_agent(demo_state)

    # 3. Print the final results
    print("\n\n--- Workflow Complete ---")
    print("Final State:")
    print(f"  Best Hyperparameters for {final_state['plan']['model_name']}:")
    for key, value in final_state['best_hyperparameters'].items():
        print(f"    {key}: {value}")
    
    print(f"\n  Forecast Result (first 10 values):")
    print(f"  {final_state['forecast_result'][:10]}")
    print("-" * 25)