# FX-Agent/src/models/model_factory.py

from torch.nn import Module
from typing import Dict, Any

# Import dependencies based on user instructions.
# This mixed import style suggests a specific project setup where both the 'src'
# directory and the root containing 'FX-Agent' are in the Python path.
from src.models.LSTM_Attention import LSTM_Attention
from FX_Agent.model.Transformer import Transformer

class ModelFactory:
    """
    A factory class for creating different model instances based on their names.
    This class is used by the HyperparameterOptimizer to instantiate models
    with dynamic parameters from Optuna trials.
    """
    @staticmethod
    def create_model(model_name: str, **kwargs: Any) -> Module:
        """
        Creates and returns a model instance based on the provided model name.

        Args:
            model_name (str): The name of the model to create.
                              Supported values are "LSTM" and "Transformer".
            **kwargs (Any): Hyperparameters for the model's constructor.

        Returns:
            Module: An instance of the requested model.

        Raises:
            ValueError: If the provided model_name is not supported.
        """
        if model_name == "LSTM":
            return LSTM_Attention(**kwargs)
        elif model_name == "Transformer":
            return Transformer(**kwargs)
        else:
            raise ValueError(f"Unknown model name provided: '{model_name}'")

if __name__ == '__main__':
    """
    Demonstration of the ModelFactory's functionality.
    This block shows how to create model instances and how the factory
    handles unknown model names.
    """
    # Define sample hyperparameters for the LSTM model
    lstm_hyperparams = {
        'input_dim': 8,
        'hidden_dim': 64,
        'n_layers': 2,
        'output_dim': 1,
        'drop_prob': 0.1
    }

    # Define sample hyperparameters for the Transformer model
    # Note: These are placeholder parameters. The actual required parameters
    # may vary depending on the specific implementation of the Transformer model.
    transformer_hyperparams = {
        'input_dim': 8,
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'output_sequence_length': 1
    }

    print("--- Demonstrating ModelFactory ---")

    # 1. Create an LSTM model instance
    print("\n1. Creating LSTM model...")
    try:
        # To run this demo script directly, the execution path needs to be
        # configured correctly to resolve imports. We assume the main application handles this.
        lstm_model = ModelFactory.create_model("LSTM", **lstm_hyperparams)
        print("   Successfully created LSTM model instance.")
        print(lstm_model)
    except ImportError as e:
        print(f"   Could not create model due to ImportError: {e}")
        print("   Please run from the correct project root directory.")
    except Exception as e:
        print(f"   Error creating LSTM model: {e}")

    # 2. Create a Transformer model instance
    print("\n2. Creating Transformer model...")
    try:
        transformer_model = ModelFactory.create_model("Transformer", **transformer_hyperparams)
        print("   Successfully created Transformer model instance.")
        print(transformer_model)
    except ImportError as e:
        print(f"   Could not create model due to ImportError: {e}")
        print("   Please run from the correct project root directory.")
    except Exception as e:
        print(f"   Error creating Transformer model: {e}")

    # 3. Test error handling for an unknown model
    print("\n3. Attempting to create an unsupported model (e.g., 'GRU')...")
    try:
        ModelFactory.create_model("GRU", **lstm_hyperparams)
    except ValueError as e:
        print(f"   Successfully caught expected error: {e}")
