# Machine Learning Model Deployment Pipeline

## Mission
EzanAi is an AI-powered app which will be helping farmers and other people in the field of agriculture get accurate information about history of crop yield for making decisions related to agricultural risk management and future predictions. Crops yield classification value in hectogram per hectare (Hg/Ha) is got in a certain year according to the crop, weather conditions(Average rain fall per year,temperature) and Pesticides used in tonnes.


## Project Overview
This project demonstrates the full lifecycle of a Machine Learning classification model, including training, evaluation, deployment, scaling, and monitoring.

### Description of dataset

The  dataset used contains agricultural yield data across various regions, crops, and years. It includes key factors such as rainfall, temperature, and pesticide usage, aiming to analyze their impact on crop yields.
[Source of dataset: Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

## Classification Categories
- **High Yield**: Indicates optimal production conditions and farming practices
- **Medium Yield**: Suggests average production with potential for improvement
- **Low Yield**: Signals need for intervention in farming practices or conditions
  
## Features
1. **Model Prediction** - Users can input features to get predictions.
2. **Data Visualization** - Meaningful visualizations of key dataset features.
3. **Bulk Data Upload** - Users can upload a CSV file with multiple rows for batch predictions.
4. **Retraining Trigger** - Users can upload new data and trigger model retraining.


## Project Structure
```
Project_name/
│── README.md
│── notebook/
│   ├── project_name.ipynb       # Jupyter Notebook for model development
│── src/
│   ├── preprocessing.py         # Data preprocessing functions
│   ├── model.py                 # Model training and evaluation
│   ├── prediction.py            # Model inference script
│── data/
│   ├── train/                   # Training dataset
│   ├── test/                    # Testing dataset
│── models/
│   ├── yieldclassifier_model.pkl            # Saved ML model 
│   ├── yieldscaler.pkl             
             # Dockerfile for containerization                  
│── requirements.txt             # Dependencies
```

##  Data Preprocessing & Model Training
### 1️. Preprocessing
- Data cleaning, handling missing values, feature selection, and encoding.
- Implemented in `src/preprocessing.py`.

### 2️. Model Training & Evaluation
- Train a classification model offline.
- Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
- Implemented in `src/model.py` and Jupyter Notebook (`notebook/project_name.ipynb`).

##  Deployment & Cloud Integration
1. **Cloud Deployment**: The model is deployed on [Cloud Platform] using `deployment/cloud_config.yml`.
2. **API Endpoint**: The model is exposed as an API for making predictions.
3. **Automated Retraining**: New data can be uploaded and used to retrain the model automatically.

## 🎯 How to Use
### 1️. Running the Model Locally
```bash
# Clone the repository
git clone https://github.com/lilika67/MLOP_summative.git
cd MLOP_summative

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```


### 3️. Making Predictions and retrain via swagger local  url 
http://localhost:8000/docs


##  How to run the FASTapi on production
To run this fastApi you can use the swagger docs through the link[ Swagger UI](https://ezanai.onrender.com/docs)

## How to run this app on frontend
To efficiently use EzanAI app we also created a user friendly web app which will be helping people to test all functionalities.
to test it use [ Frontend link](https://ezanai.onrender.com/docs)


