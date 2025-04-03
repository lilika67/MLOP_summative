from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  
from src.preprocessing import CropData, preprocess_single_data, preprocess_mongodb_data, load_test_data
from src.model import load_model, retrain_model, predict, evaluate_model
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

app = FastAPI(title="Crop Yield Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

if not MONGO_URI or not MONGO_DB or not MONGO_COLLECTION:
    raise ValueError("One or more MongoDB environment variables are missing. Check your .env file.")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

model = load_model()

@app.post("/predict/")
async def predict_yield(data: CropData):
    try:
        X_scaled = preprocess_single_data(data)
        prediction = predict(model, X_scaled)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/retrain/")
async def retrain():
    try:
        X_scaled, y = preprocess_mongodb_data()
        global model
        model, retrained_model_path = retrain_model(model, X_scaled, y)
        X_test_scaled, y_test = load_test_data()
        metrics = evaluate_model(model, X_test_scaled, y_test)
        return {
            "message": f"Model retrained successfully and saved as '{retrained_model_path}'",
            "metrics": metrics
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.post("/upload/")
async def upload_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        if 'yield_category' not in df.columns:
            raise ValueError("Uploaded CSV must contain 'yield_category' column")
        
        data = df.to_dict('records')
        collection.drop()
        collection.insert_many(data)
        
        return {"message": f"Successfully uploaded {len(data)} records to MongoDB collection '{MONGO_COLLECTION}'"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")
