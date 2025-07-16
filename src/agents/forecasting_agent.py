import pandas as pd
from ..core.abstractions.base_agent import BaseAgent
from ..core.workflow_state import AppState
from ..services.forecasting_service import MockForecastingService

class ForecastingAgent(BaseAgent):
    def __init__(self, forecasting_service: MockForecastingService):
        self._forecasting_service = forecasting_service

    def run(self, state: AppState) -> AppState:
        print("--- Forecasting Agent Running ---")
        
        # Mock data creation
        mock_df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8]
        })
        
        params = {"model_name": "TFT"} # From config in a real scenario
        
        forecast = self._forecasting_service.predict(mock_df, params)
        
        state.forecast_result = forecast
        state.next_step = "END" # Signal the end of the workflow
        
        print(f"Forecast generated: {state.forecast_result}")
        return state