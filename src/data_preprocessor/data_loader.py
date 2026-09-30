from src.logger import logger
import pandas as pd

class DataLoader():
    def __init__(self, path):
        self.path=path

    def load_data(self):
        """
        Loads the raw csv file and returns a pandas Dataframe
        
        Parameters
        ----------
        path: data path

        Returns
        -------
        df: pandas.Dataframe
            loaded PM2 Readings
     
        """
        logger.info(f"Loading raw data from {self.path}")
        try:
            df=pd.read_csv(self.path)
            return df
        except Exception as e:
            logger.error(f"failed to load data from {self.path}: {e}")
            raise

    def wrangle(self, df):
        """
        Cleans and transformsthe loaded dataframe and returns a dataframe suitable for modelling
         
        Parameters
        ----------
        df: pandas.Dataframe
            loaded PM2 readings

        Returns
        -------
        df: pandas.Dataframe
            clean dataframe ready for modelling
        """
        logger.info(f"Starting data cleaning")
        try:
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
        
            # Turn index to datetime
            df.index = pd.to_datetime(df.index)
        
            # Convert to local timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert("Africa/Nairobi")

            # Resample readings to hourly windows and forwad fill null values
            df=df["P2"].resample("1h").mean().ffill().to_frame()

            # Add lag feature
            df["P2.L1"]=df["P2"].shift(1)
            df=df.dropna()

            return df
            
        except Exception as e:
            logger.error(f"Failed to wrangle data: {e}")
            raise


    