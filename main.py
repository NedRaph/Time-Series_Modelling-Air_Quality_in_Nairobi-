from src.data_preprocessor.data_loader import DataLoader
from src.config import RAW_DATA_PATH, MODEL_PATH
from src.modelling.linear_model import Modeler
from src.modelling.model_manager import ModelManager
from src.logger import logger

def main():
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    df = loader.wrangle(df)
    modeler = Modeler(df)
    model, mae = modeler.train_model() 
    model_manager = ModelManager(MODEL_PATH)
    model_manager.save_model(model)
    logger.info(f"Model training and saving completed. Training MAE: {mae:.2f}")

if __name__ == "__main__":
    main()


