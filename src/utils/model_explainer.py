import pandas as pd
import time

def explain_model(model, test_data: pd.DataFrame) -> dict:
    """
    Simulates the process of model explanation using a tool like SHAP.

    In a real-world scenario, this function would use SHAP to compute feature
    importance values. Here, we simulate the process by returning a fixed
    dictionary of feature importances based on the columns of the test data.

    Args:
        model: A trained model object (its type is not critical for this simulation).
        test_data (pd.DataFrame): The test dataset, used to identify feature names.

    Returns:
        dict: A dictionary mapping feature names to their simulated importance scores.
    """
    print("\nStarting model explanation simulation (SHAP)...")
    print(f"Generating feature importances for {len(test_data.columns)} features.")

    # Simulate the time taken for explanation
    time.sleep(3)

    # Create a mock feature importance dictionary
    # In a real scenario, these values would be calculated by SHAP.
    # We'll assign some arbitrary decreasing values for demonstration.
    feature_importances = {
        feature: round(0.8 / (i + 1), 3) 
        for i, feature in enumerate(test_data.columns)
    }
    
    # Normalize to sum to 1 (optional, but good practice)
    total_importance = sum(feature_importances.values())
    if total_importance > 0:
        feature_importances = {k: round(v / total_importance, 3) for k, v in feature_importances.items()}


    print("Model explanation simulation finished.")
    print(f"Feature importances: {feature_importances}")

    return feature_importances
