from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseForecasting(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a forecast based on the input data and parameters."""
        pass