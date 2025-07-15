import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# PART 1: ALGORITHM FRAMEWORK DEFINITION
# ============================================================================

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        skip = x
        if self.skip_proj:
            skip = self.skip_proj(skip)

        x = self.input_proj(x)
        if context is not None and self.context_proj is not None:
            x += self.context_proj(context)
        
        x = torch.relu(x)
        x = self.hidden_layer(x)
        g = self.gate(x)
        x = self.dropout(x * g)
        x = self.output_proj(x)
        
        return self.layer_norm(x + skip)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_inputs, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        self.grn = GatedResidualNetwork(input_dim * num_inputs, hidden_dim, num_inputs, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)
        weights = self.grn(flat_x)
        weights = self.softmax(weights).unsqueeze(1)
        
        return weights.view(x.size(0), 1, self.num_inputs, 1)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_static_inputs, num_past_inputs, num_future_inputs, 
                 sequence_length, horizon, output_quantiles, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.num_static_inputs = num_static_inputs
        self.num_past_inputs = num_past_inputs
        self.num_future_inputs = num_future_inputs
        self.output_quantiles = output_quantiles

        self.static_vsn = VariableSelectionNetwork(1, hidden_dim, num_static_inputs)
        self.past_vsn = VariableSelectionNetwork(1, hidden_dim, num_past_inputs)
        self.future_vsn = VariableSelectionNetwork(1, hidden_dim, num_future_inputs)

        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        self.lstm_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.decoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        self.output_proj = nn.Linear(hidden_dim * horizon, horizon * len(output_quantiles))

    def forward(self, x_static, x_past, x_future):
        static_weights = self.static_vsn(x_static.unsqueeze(-1))
        static_embedding = torch.sum(static_weights * x_static.unsqueeze(1).unsqueeze(-1), dim=2).squeeze(-1)
        
        past_weights = self.past_vsn(x_past.unsqueeze(-1))
        past_embedding = torch.sum(past_weights * x_past.unsqueeze(1).unsqueeze(-1), dim=2).squeeze(-1)

        future_weights = self.future_vsn(x_future.unsqueeze(-1))
        future_embedding = torch.sum(future_weights * x_future.unsqueeze(1).unsqueeze(-1), dim=2).squeeze(-1)
        
        static_context = self.static_enrichment(static_embedding)
        
        temporal_input = torch.cat([past_embedding, future_embedding], dim=1)
        enriched_input = temporal_input + static_context.unsqueeze(1)
        
        lstm_out, _ = self.lstm_encoder(enriched_input)
        
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_gate(attn_out)
        attn_out = self.attn_norm(attn_out + lstm_out)
        
        decoder_out = self.decoder_grn(attn_out[:, self.sequence_length:, :])
        
        output = decoder_out.reshape(decoder_out.size(0), -1)
        output = self.output_proj(output)
        
        return output.view(output.size(0), self.horizon, len(self.output_quantiles))

def prepare_tft_data(data, static_cols, past_cols, future_cols, target, sequence_length, horizon):
    data_list = []
    for i in range(len(data) - sequence_length - horizon + 1):
        past_end = i + sequence_length
        future_end = past_end + horizon
        
        static_features = data[static_cols].iloc[i].values
        past_features = data[past_cols].iloc[i:past_end].values
        future_features = data[future_cols].iloc[past_end:future_end].values
        target_values = data[target].iloc[past_end:future_end].values
        
        data_list.append((static_features, past_features, future_features, target_values))
    
    return data_list

def train_tft_model(data, model, optimizer, loss_fn, device, batch_size=64):
    model.train()
    total_loss = 0
    np.random.shuffle(data)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        x_static = torch.FloatTensor(np.array([item[0] for item in batch])).to(device)
        x_past = torch.FloatTensor(np.array([item[1] for item in batch])).to(device)
        x_future = torch.FloatTensor(np.array([item[2] for item in batch])).to(device)
        y_true = torch.FloatTensor(np.array([item[3] for item in batch])).to(device)
        
        optimizer.zero_grad()
        y_pred = model(x_static, x_past, x_future)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (len(data) / batch_size)

def evaluate_tft_model(data, model, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for item in data:
            x_static = torch.FloatTensor(item[0]).unsqueeze(0).to(device)
            x_past = torch.FloatTensor(item[1]).unsqueeze(0).to(device)
            x_future = torch.FloatTensor(item[2]).unsqueeze(0).to(device)
            y_true = torch.FloatTensor(item[3]).unsqueeze(0).to(device)
            
            y_pred = model(x_static, x_past, x_future)
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_true.cpu().numpy())

    return total_loss / len(data), np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)

def plot_tft_predictions(preds, trues, quantile_idx_p50, quantile_idx_lower, quantile_idx_upper, num_to_plot=100):
    plt.figure(figsize=(15, 7))
    plt.plot(trues[:num_to_plot, 0], 'b-', label='Actual')
    plt.plot(preds[:num_to_plot, 0, quantile_idx_p50], 'r-', label='P50 Forecast')
    plt.fill_between(
        np.arange(num_to_plot),
        preds[:num_to_plot, 0, quantile_idx_lower],
        preds[:num_to_plot, 0, quantile_idx_upper],
        color='red', alpha=0.2, label='P10-P90 Range'
    )
    plt.title('TFT Forecast vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
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
    # The dataframe 'df' must be prepared by you.
    # df = pd.read_csv('your_data_file.csv') 
    print("WARNING: Using placeholder data. You must load your own data for a real use case.")
    df = pd.DataFrame({
        'static_feature_1': np.ones(200),
        'past_feature_1': np.random.rand(200),
        'future_feature_1': np.random.rand(200),
        'target_column': np.random.rand(200)
    })


    # --- 3. DEFINE ALL PARAMETERS - YOU MUST FILL THESE IN ---
    
    # Data columns
    STATIC_COLS = [...]
    PAST_COLS = [...]
    FUTURE_COLS = [...]
    TARGET = '...'

    # Data parameters
    SEQUENCE_LENGTH = ... # Example: 60 
    HORIZON = ...         # Example: 20
    
    # Model hyperparameters
    HIDDEN_DIM = ...        # Example: 32
    NUM_HEADS = ...         # Example: 4
    OUTPUT_QUANTILES = [...] # Example: [0.1, 0.5, 0.9]
    EPOCHS = ...            # Example: 100

    # --- 4. Prepare Data for TFT ---
    # NOTE: You should apply scaling (e.g., MinMaxScaler) to your data columns before this step.
    
    # This block will raise an error until you fill in the parameters above.
    try:
        full_data = prepare_tft_data(df, STATIC_COLS, PAST_COLS, FUTURE_COLS, TARGET, SEQUENCE_LENGTH, HORIZON)
    except (NameError, TypeError):
        print("\nERROR: Please fill in the placeholder parameters (...) in section 3 before running.")
        exit()

    train_size = int(len(full_data) * 0.8)
    train_data = full_data[:train_size]
    test_data = full_data[train_size:]

    print(f"Total samples: {len(full_data)}, Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

    # --- 5. Initialize Model, Loss, and Optimizer ---
    model = TemporalFusionTransformer(
        num_static_inputs=len(STATIC_COLS),
        num_past_inputs=len(PAST_COLS),
        num_future_inputs=len(FUTURE_COLS),
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        output_quantiles=OUTPUT_QUANTILES,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS
    ).to(device)
    
    loss_fn = QuantileLoss(quantiles=OUTPUT_QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # --- 6. Train the Model ---
    print("Starting model training...")
    for epoch in range(EPOCHS):
        train_loss = train_tft_model(train_data, model, optimizer, loss_fn, device)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}')

    # --- 7. Evaluate and Plot Results ---
    print("\nEvaluating model...")
    test_loss, preds, trues = evaluate_tft_model(test_data, model, loss_fn, device)
    print(f'Test Loss: {test_loss:.4f}')

    plot_tft_predictions(
        preds=preds,
        trues=trues,
        quantile_idx_p50=OUTPUT_QUANTILES.index(0.5), # Assumes 0.5 is in your quantiles
        quantile_idx_lower=0, # Index of the lowest quantile
        quantile_idx_upper=-1, # Index of the highest quantile
        num_to_plot=min(100, len(test_data))
    )