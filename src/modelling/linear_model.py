import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from src.logger import logger

class Modeler(self, df):
    def __init__(self, df):
        self.df=df

    def train_model(self):
        """
        Trains a linear regression model on the cleaned dataframe and returns the trained model and its performance metrics
         
        Parameters
        ----------
        df: pandas.Dataframe
            clean dataframe ready for modelling

        Returns
        -------
        model: sklearn.linear_model.LinearRegression
            trained linear regression model
        mae: float
            mean absolute error of the model on the training data
        """
        logger.info(f"Starting model training")
        try:
            # Split data into features and target variable
            X = self.df[["P2.L1"]]
            y = self.df["P2"]

            # Initialize and train the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Make predictions on the training data
            y_pred = model.predict(X)

            # Calculate mean absolute error
            mae = mean_absolute_error(y, y_pred)

            logger.info(f"Model training completed with MAE: {mae}")
            return model, mae

        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    