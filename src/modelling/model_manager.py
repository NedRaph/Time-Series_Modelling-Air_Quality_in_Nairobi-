import joblib
from src.logger import logger
from src.config import MODEL_PATH


class ModelManager:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path=model_path

    def save_model(self, model):
        """
        Saves the trained model to disk using joblib.

        Parameters
        ----------
        model: sklearn.linear_model.LinearRegression
            trained linear regression model
        """
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved successfully at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self):
        """
        Loads a trained model from disk using joblib.
    
        Returns
        -------
        model: sklearn.linear_model.LinearRegression
            loaded linear regression model
        """
        try:
            self.model=joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise