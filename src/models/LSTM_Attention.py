import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ============================================================================
# PART 1: ALGORITHM FRAMEWORK DEFINITION
# (The core, reusable implementation of the model and its functions)
# ============================================================================

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.attention_vector = nn.Parameter(torch.rand(hidden_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn(lstm_output))
        attn_scores = torch.einsum("bsh,h->bs", energy, self.attention_vector)
        attention_weights = self.softmax(attn_scores)
        context_vector = torch.einsum("bs,bsh->bh", attention_weights, lstm_output)
        return context_vector

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        output = self.output_layer(context_vector)
        return output

def prepare_data(data, features, target, test_size=0.2, sequence_length=10):
    df_cleaned = data.dropna(subset=features + [target])
    X = df_cleaned[features]
    y = df_cleaned[target]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, shuffle=False
    )
    
    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_train).squeeze(),
        torch.FloatTensor(y_test).squeeze(),
        scaler_y,
        scaler_X
    )

def train_model(X_train, y_train, input_dim, hidden_dim, num_layers, output_dim, dropout, device, epochs=300):
    model = LSTMAttentionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    model.train()
    train_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        outputs = model(X_train_gpu)
        loss = criterion(outputs.squeeze(), y_train_gpu)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        train_losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
    return model, train_losses

def evaluate_model(model, X_test, y_test, scaler_y, device='cuda'):
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test.to(device)
        predictions = model(X_test_gpu)
    
    predictions_np = predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    y_test_orig = scaler_y.inverse_transform(y_test_np.reshape(-1, 1))
    predictions_orig = scaler_y.inverse_transform(predictions_np.reshape(-1, 1))
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
    mape = mean_absolute_percentage_error(y_test_orig, predictions_orig)
    
    return rmse, mape, y_test_orig, predictions_orig

def plot_predictions(y_test_orig, predictions_orig, title='Prediction vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig, label='Actual Values', color='blue', alpha=0.8)
    plt.plot(predictions_orig, label='Predicted Values', color='red', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 2: USER CONFIGURATION & EXECUTION TEMPLATE
# (Fill in the parameters below to run the model with your own data)
# ============================================================================

if __name__ == '__main__':
    
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load and Prepare Your Data ---
    # Replace with the path to your data file.
    # df = pd.read_csv('your_data_file.csv')
    # Ensure your DataFrame is correctly formatted, with feature columns and a target column.
    # For this template, we create a placeholder DataFrame.
    # You MUST replace this with your actual data loading logic.
    placeholder_data = {
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'your_target': np.random.rand(100)
    }
    df = pd.DataFrame(placeholder_data)


    # --- 3. Define Data and Model Parameters ---
    
    # Data processing parameters
    # IMPORTANT: Define your feature columns and target column.
    # Example: FEATURES = ['open', 'high', 'low', 'volume']
    FEATURES = ['feature_1', 'feature_2']
    # Example: TARGET = 'close'
    TARGET = 'your_target'
    # Sequence length for the LSTM model
    SEQUENCE_LENGTH = 15
    TEST_SIZE = 0.2

    # Model hyperparameters
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.1
    EPOCHS = 100
    
    # These are typically derived from the data parameters
    INPUT_DIM = len(FEATURES)
    OUTPUT_DIM = 1

    # --- 4. Run the Full Pipeline ---
    
    # Prepare the data for the model
    X_train, X_test, y_train, y_test, scaler_y, scaler_X = prepare_data(
        data=df,
        features=FEATURES,
        target=TARGET,
        test_size=TEST_SIZE,
        sequence_length=SEQUENCE_LENGTH
    )
    
    print("Data prepared successfully.")
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

    # Train the model
    trained_model, losses = train_model(
        X_train, y_train,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        device=device,
        epochs=EPOCHS
    )

    # Evaluate the model on the test set
    rmse, mape, y_test_orig, predictions_orig = evaluate_model(
        trained_model, X_test, y_test, scaler_y, device
    )
    
    print(f'\nTest Set Evaluation Results:')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.4f}')

    # Visualize the predictions
    plot_predictions(y_test_orig, predictions_orig)