import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ============================================================================
# PART 1: ALGORITHM FRAMEWORK DEFINITION
# ============================================================================

class FrequencyLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, d_model))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        fft_x = torch.fft.rfft(x, dim=1)
        
        weighted_real = F.linear(fft_x.real, self.weight)
        weighted_imag = F.linear(fft_x.imag, self.weight)
        
        weighted_fft = torch.complex(weighted_real, weighted_imag)
        x_reconstructed = torch.fft.irfft(weighted_fft, dim=1, n=seq_len)
        return x_reconstructed

class TimesNetBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.freq_layer = FrequencyLayer(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.freq_layer(x)
        x = self.norm1(x + residual)
        x = self.dropout1(x)
        
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        x = self.dropout2(x)
        return x

class TimesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.timesnet_blocks = nn.ModuleList([
            TimesNetBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.timesnet_blocks:
            x = block(x)
        output = self.output_layer(x[:, -1, :])
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
        torch.FloatTensor(y_train),
        torch.FloatTensor(y_test),
        scaler_y,
        scaler_X
    )

def train_model(X_train, y_train, input_dim, hidden_dim, num_layers, output_dim, dropout, device, epochs=300):
    model = TimesNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        
        outputs = model(X_train_gpu)
        loss = criterion(outputs, y_train_gpu)
        loss.backward()
        optimizer.step()
        
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
    
    y_test_orig = scaler_y.inverse_transform(y_test_np)
    predictions_orig = scaler_y.inverse_transform(predictions_np)
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
    mape = mean_absolute_percentage_error(y_test_orig, predictions_orig)
    
    return rmse, mape, y_test_orig, predictions_orig

def plot_predictions(y_true, y_pred, title='Prediction vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# ============================================================================
# PART 2: USER CONFIGURATION & EXECUTION TEMPLATE
# ============================================================================

if __name__ == '__main__':
    
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load Your Data ---
    # YOU MUST PROVIDE YOUR OWN DATA LOADING LOGIC HERE.
    # df = pd.read_csv('your_data_file.csv') 
    print("WARNING: Using placeholder data. You must load your own data for a real use case.")
    df = pd.DataFrame({
        'feature1': np.random.rand(200),
        'feature2': np.random.rand(200),
        'target_column': np.random.rand(200)
    })

    # --- 3. DEFINE ALL PARAMETERS - YOU MUST FILL THESE IN ---
    
    # Data processing parameters
    FEATURES = [...]         # Example: ['feature1', 'feature2']
    TARGET = '...'           # Example: 'target_column'
    SEQUENCE_LENGTH = ...    # Example: 30
    TEST_SIZE = ...          # Example: 0.2

    # Model hyperparameters
    HIDDEN_DIM = ...         # Example: 64
    NUM_LAYERS = ...         # Example: 3
    DROPOUT = ...            # Example: 0.1
    EPOCHS = ...             # Example: 150
    
    # These are derived automatically
    try:
        INPUT_DIM = len(FEATURES)
    except TypeError:
        INPUT_DIM = 0 
    OUTPUT_DIM = 1
    
    # --- 4. Run the Full Pipeline ---
    try:
        if '...' in [TARGET] or ... in [FEATURES, SEQUENCE_LENGTH, TEST_SIZE, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EPOCHS]:
             raise ValueError("Parameters not defined.")

        print("Preparing data...")
        X_train, X_test, y_train, y_test, scaler_y, scaler_X = prepare_data(
            data=df,
            features=FEATURES,
            target=TARGET,
            test_size=TEST_SIZE,
            sequence_length=SEQUENCE_LENGTH
        )
        
        print("Starting model training...")
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

        print("Evaluating model...")
        rmse, mape, y_test_orig, predictions_orig = evaluate_model(
            trained_model, X_test, y_test, scaler_y, device
        )
        
        print(f'\nTest Set Evaluation Results:')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAPE: {mape:.4f}')

        print("Plotting results...")
        plot_predictions(y_test_orig, predictions_orig)

    except (NameError, TypeError, ValueError):
        print("\nERROR: Please fill in all the placeholder parameters (...) in section 3 before running the script.")