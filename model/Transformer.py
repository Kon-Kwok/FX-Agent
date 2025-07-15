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
# ============================================================================

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x[:, -1, :])
        return x

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

def train_model(X_train, y_train, input_dim, hidden_dim, num_layers, num_heads, output_dim, device, epochs=300):
    model = TransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=output_dim
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        
        outputs = model(X_train_gpu)
        loss = criterion(outputs, y_train_gpu)
        
        if torch.isnan(loss):
            print(f"Epoch {epoch+1}: Loss is NaN. Stopping training.")
            break
        
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
    
    y_test_orig = scaler_y.inverse_transform(y_test_np)
    predictions_orig = scaler_y.inverse_transform(predictions_np)
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
    mape = mean_absolute_percentage_error(y_test_orig, predictions_orig)
    
    return rmse, mape, y_test_orig, predictions_orig

def plot_predictions(y_test_orig, predictions_orig, title='Prediction vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig, label='Actual Values', color='blue')
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
# ============================================================================

if __name__ == '__main__':
    
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load and Prepare Your Data ---
    # TODO: Replace this with your actual data loading logic.
    # df = pd.read_csv('your_data_file.csv')
    placeholder_data = {
        'feature1': np.random.rand(200),
        'feature2': np.random.rand(200),
        'target_variable': np.random.rand(200)
    }
    df = pd.DataFrame(placeholder_data)

    # --- 3. Define Data and Model Parameters ---
    
    # Data processing parameters
    FEATURES = ['feature1', 'feature2']  # TODO: Define your list of feature columns
    TARGET = 'target_variable'           # TODO: Define your target column name
    SEQUENCE_LENGTH = 30                 # TODO: Define the sequence length for the Transformer
    TEST_SIZE = 0.2                      # TODO: Define the test set size

    # Model hyperparameters
    HIDDEN_DIM = 128                     # TODO: Define the model's hidden dimension (d_model)
    NUM_LAYERS = 3                       # TODO: Define the number of Transformer Encoder layers
    NUM_HEADS = 8                        # TODO: Define the number of attention heads (must be a divisor of HIDDEN_DIM)
    EPOCHS = 150                         # TODO: Define the number of training epochs
    
    # These are derived automatically from the parameters above
    INPUT_DIM = len(FEATURES)
    OUTPUT_DIM = 1

    # --- 4. Run the Full Pipeline ---
    
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
        num_heads=NUM_HEADS,
        output_dim=OUTPUT_DIM,
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