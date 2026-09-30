from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent

RAW_DATA_PATH=BASE_DIR/"data"/"clean_nairobi_p2_readings.csv"
MODEL_PATH=BASE_DIR/"artifacts"/"model.pkl"