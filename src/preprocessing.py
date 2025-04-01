import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

SCALER_PATH = "models/yieldscaler.pkl"
scaler = joblib.load(SCALER_PATH)

TRAIN_DATA_PATH = "data/train/cropyield_train.csv"
train_df = pd.read_csv(TRAIN_DATA_PATH)

FEATURE_COLUMNS = [col for col in train_df.columns if col != 'yield_category']

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


if not MONGO_URI or not MONGO_DB or not MONGO_COLLECTION:
    raise ValueError("One or more MongoDB environment variables are missing. Check your .env file.")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

class CropData(BaseModel):
    Area: str = Field(..., description="Country or region of the crop.")
    Item: str = Field(
        ...,
        description="Crop type. Must be one of: Maize, Potatoes, Rice, paddy, Sorghum, Soybeans, Wheat, Sweet potatoes, Plantains, Yams.",
        enum=["Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Wheat", "Sweet potatoes", "Plantains", "Yams"]
    )
    Year: int = Field(
        ...,
        ge=1990, le=2050,
        description="Year of prediction. Must be between 1990 and 2050."
    )
    average_rain_fall_mm_per_year: float = Field(
        ...,
        ge=51, le=3240,
        description="Average annual rainfall in mm. Must be between 51 and 3240."
    )
    pesticides_tonnes: float = Field(
        ...,
        ge=0.04, le=36778,
        description="Pesticides used in tonnes. Must be between 0.04 and 36778."
    )
    avg_temp: float = Field(
        ...,
        ge=1.3, le=30.65,
        description="Average temperature in Celsius. Must be between 1.3 and 30.65."
    )

def preprocess_single_data(data: CropData) -> np.ndarray:
    df = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df['Year'] = data.Year
    df['average_rain_fall_mm_per_year'] = data.average_rain_fall_mm_per_year
    df['pesticides_tonnes'] = data.pesticides_tonnes
    df['avg_temp'] = data.avg_temp
    
    area_col = f'Area_{data.Area}'
    if area_col in FEATURE_COLUMNS:
        df[area_col] = 1
    else:
        raise ValueError(f"Unknown Area: {data.Area}. Must be one of {[col.replace('Area_', '') for col in FEATURE_COLUMNS if col.startswith('Area_')]}")
    
    item_col = f'Item_{data.Item}'
    if item_col in FEATURE_COLUMNS:
        df[item_col] = 1
    else:
        raise ValueError(f"Unknown Item: {data.Item}. Must be one of {[col.replace('Item_', '') for col in FEATURE_COLUMNS if col.startswith('Item_')]}")
    
    X_scaled = scaler.transform(df)
    return X_scaled

def preprocess_mongodb_data() -> tuple:
    data = list(collection.find())
    if not data:
        raise ValueError(f"No data found in MongoDB collection '{MONGO_COLLECTION}'")
    
    df = pd.DataFrame(data)
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    
    if 'yield_category' not in df.columns:
        raise ValueError("MongoDB data must contain 'yield_category' column")
    
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        for col in missing_cols:
            df[col] = 0
    
    X = df[FEATURE_COLUMNS]
    y = df['yield_category']
    X_scaled = scaler.transform(X)
    return X_scaled, y

def load_test_data():
    test_df = pd.read_csv("data/test/cropyield_test.csv")
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['yield_category']
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test