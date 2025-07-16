import pandas as pd
from typing import Dict, Any
from ..core.abstractions.base_forecasting import BaseForecasting

class MockForecastingService(BaseForecasting):
    """A mock forecasting service."""
    def predict(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"--- MockForecastingService Predicting ---")
        print(f"Data shape: {data.shape}, Params: {params}")
        return {"forecast": [1, 2, 3], "confidence": 0.9}