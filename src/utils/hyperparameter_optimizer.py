import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

# Assuming ModelFactory exists in src.models.model_factory
# This import will work after it's created in a subsequent task
from src.models.model_factory import ModelFactory

class HyperparameterOptimizer:
    """
    A class for hyperparameter optimization of PyTorch models using Optuna.
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Initializes the optimizer.

        Args:
            X_train (np.ndarray): Features of the training set.
            y_train (np.ndarray): Target of the training set.
            X_test (np.ndarray): Features of the test set.
            y_test (np.ndarray): Target of the test set.
        """
        self.X_train_tensor = torch.from_numpy(X_train).float()
        self.y_train_tensor = torch.from_numpy(y_train).float()
        self.X_test_tensor = torch.from_numpy(X_test).float()
        self.y_test_tensor = torch.from_numpy(y_test).float()

    def _objective(self, trial: optuna.trial.Trial, model_name: str) -> float:
        """
        The objective function for Optuna to evaluate a set of hyperparameters.

        Args:
            trial (optuna.trial.Trial): An Optuna Trial object for suggesting hyperparameters.
            model_name (str): The name of the model to be optimized.

        Returns:
            float: The loss value calculated on the test set.
        """
        # 1. Define the hyperparameter search space
        params = {
            "input_dim": self.X_train_tensor.shape[2],
            "output_dim": self.y_train_tensor.shape[1],
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5)
        }
        
        # 2. Instantiate the model
        model = ModelFactory.create_model(model_name, **params)
        
        # 3. Train and evaluate
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.MSELoss()
        
        # Training loop
        num_epochs = 50  # Set a small number of epochs for quick evaluation
        batch_size = 32
        
        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(self.X_train_tensor), batch_size):
                X_batch = self.X_train_tensor[i:i+batch_size]
                y_batch = self.y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(self.X_test_tensor)
            test_loss = criterion(test_outputs, self.y_test_tensor)
            
        return test_loss.item()

    def optimize(self, model_name: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Executes the hyperparameter optimization.

        Args:
            model_name (str): The name of the model to optimize (e.g., "LSTM", "RNN").
            n_trials (int): The number of trials for the Optuna optimization.

        Returns:
            Dict[str, Any]: A dictionary of the best hyperparameters found.
        """
        study = optuna.create_study(direction="minimize")
        
        # Use a lambda function to pass model_name
        objective_func = lambda trial: self._objective(trial, model_name)
        
        study.optimize(objective_func, n_trials=n_trials)
        
        print(f"Optimization finished. Best trial loss: {study.best_value}")
        print("Best parameters found: ", study.best_trial.params)
        
        return study.best_trial.params

if __name__ == '__main__':
    # This is a demonstration and requires a functional ModelFactory and model implementation to run.
    
    # 1. Mock ModelFactory and LSTM model
    class MockLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            h_lstm, _ = self.lstm(x)
            return self.fc(h_lstm[:, -1, :])

    class ModelFactory:
        @staticmethod
        def create_model(model_name: str, **kwargs: Any) -> nn.Module:
            if model_name.upper() == "LSTM":
                # Filter out parameters that do not belong to MockLSTM
                lstm_params = {
                    "input_dim": kwargs["input_dim"],
                    "hidden_dim": kwargs["hidden_dim"],
                    "num_layers": kwargs["num_layers"],
                    "output_dim": kwargs["output_dim"],
                    "dropout": kwargs["dropout"]
                }
                return MockLSTM(**lstm_params)
            else:
                raise ValueError(f"Model {model_name} not supported by MockFactory.")

    # 2. Create mock data
    # (n_samples, sequence_length, n_features)
    X_train_np = np.random.rand(100, 10, 5).astype(np.float32)
    y_train_np = np.random.rand(100, 1).astype(np.float32)
    X_test_np = np.random.rand(50, 10, 5).astype(np.float32)
    y_test_np = np.random.rand(50, 1).astype(np.float32)

    # 3. Instantiate and run the optimizer
    optimizer = HyperparameterOptimizer(
        X_train=X_train_np,
        y_train=y_train_np,
        X_test=X_test_np,
        y_test=y_test_np
    )

    # 4. Find the best parameters for the "LSTM" model
    print("Starting hyperparameter optimization for LSTM model...")
    best_params = optimizer.optimize(model_name="LSTM", n_trials=20) # Use fewer n_trials for demonstration

    print("\n--- Optimization Complete ---")
    print(f"Best hyperparameters for LSTM: {best_params}")
