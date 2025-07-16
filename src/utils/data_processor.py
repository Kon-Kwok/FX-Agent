import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict

class DataProcessor:
    """
    A class to process time-series data for machine learning models.
    It handles cleaning, feature engineering, scaling, and sequence creation.
    """
    def __init__(self, sequence_length: int = 60, test_size: float = 0.2):
        """
        Initializes the DataProcessor.

        Args:
            sequence_length (int): The length of the sequences for time-series prediction.
            test_size (float): The proportion of the dataset to allocate to the test split.
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences from the time-series data.

        Args:
            data (np.ndarray): The input data array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the features (X) and target (y) arrays.
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), :-1])
            y.append(data[i + self.sequence_length, -1])
        return np.array(X), np.array(y)

    def process_data(self, df: pd.DataFrame, target_column: str, feature_columns: List[str]) -> Dict:
        """
        Processes the raw data frame into training and testing sets.

        Args:
            df (pd.DataFrame): The input data frame with time-series data.
            target_column (str): The name of the target variable column.
            feature_columns (List[str]): A list of column names to be used as features.

        Returns:
            Dict: A dictionary containing X_train, y_train, X_test, y_test, and the scaler object.
        """
        # 1. Data Cleaning: Handle missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True) # Also backfill to handle NaNs at the start

        # 2. Feature Engineering (Simulated)
        for feature in feature_columns:
            if feature not in df.columns:
                # Simulate feature if it doesn't exist
                df[feature] = np.random.rand(len(df)) * 100

        # Ensure target_column is last for easier slicing in _create_sequences
        all_cols = feature_columns + [target_column]
        data_subset = df[all_cols].copy()

        # 3. Data Scaling
        scaled_data = self.scaler.fit_transform(data_subset)

        # 4. Create Time-Series Samples
        X, y = self._create_sequences(scaled_data)

        # 5. Split Data
        split_index = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "scaler": self.scaler
        }

if __name__ == '__main__':
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Please run: pip install yfinance")
        exit()

    # 1. Create a sample DataFrame using yfinance
    # Fetch historical data for EUR/USD
    sample_df = yf.download('EURUSD=X', start='2022-01-01', end='2023-01-01')
    
    if sample_df.empty:
        print("Failed to download data. Creating a dummy dataframe.")
        date_rng = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
        sample_df = pd.DataFrame(date_rng, columns=['Date'])
        sample_df['Open'] = np.random.uniform(1.0, 1.2, size=(len(date_rng)))
        sample_df['High'] = sample_df['Open'] + np.random.uniform(0, 0.05, size=(len(date_rng)))
        sample_df['Low'] = sample_df['Open'] - np.random.uniform(0, 0.05, size=(len(date_rng)))
        sample_df['Close'] = np.random.uniform(sample_df['Low'], sample_df['High'], size=(len(date_rng)))
        sample_df['Volume'] = np.random.randint(10000, 100000, size=(len(date_rng)))
        sample_df.set_index('Date', inplace=True)


    # 2. Instantiate the DataProcessor
    processor = DataProcessor(sequence_length=30, test_size=0.25)

    # 3. Define target and feature columns
    target = 'Close'
    # Include simulated features
    features = ['Open', 'High', 'Low', 'Volume', 'interest_rate_indicator', 'news_sentiment_score']

    # 4. Process the data
    processed_data = processor.process_data(sample_df, target_column=target, feature_columns=features)

    # 5. Print shapes to verify
    print("Data processing complete. Shapes of the resulting arrays:")
    print(f"X_train shape: {processed_data['X_train'].shape}")
    print(f"y_train shape: {processed_data['y_train'].shape}")
    print(f"X_test shape: {processed_data['X_test'].shape}")
    print(f"y_test shape: {processed_data['y_test'].shape}")

    # Verification checks
    # Expected X shape: (num_samples, sequence_length, num_features)
    # Expected y shape: (num_samples,)
    assert processed_data['X_train'].shape[1] == 30, "Sequence length mismatch in X_train"
    assert processed_data['X_test'].shape[1] == 30, "Sequence length mismatch in X_test"
    assert processed_data['X_train'].shape[2] == len(features), "Number of features mismatch in X_train"
    assert len(processed_data['X_train']) == len(processed_data['y_train']), "Train data/label mismatch"
    assert len(processed_data['X_test']) == len(processed_data['y_test']), "Test data/label mismatch"
    print("\nVerification successful!")