# Machine Learning Model Deployment Pipeline

## EzanAi Videeo Demo

To see the EzanAi app demo u can use open[Video demo](https://www.loom.com/share/29fa9468553146b6a9c1810ace51bd49?sid=d26f39b7-d4d7-421d-ac1c-ee8884bae1e9)

## Mission
EzanAi is an AI-powered app which will be helping farmers and other people in the field of agriculture get accurate information about history of crop yield for making decisions related to agricultural risk management and future predictions. Crops yield classification value in hectogram per hectare (Hg/Ha) is got in a certain year according to the crop, weather conditions(Average rain fall per year,temperature) and Pesticides used in tonnes.

## Project Overview
This project demonstrates the full lifecycle of the EzanAi Machine Learning classification model, including training, evaluation, deployment, scaling, and monitoring.

### Description of dataset

The  dataset used contains agricultural yield data across various regions, crops, and years. It includes key factors such as rainfall, temperature, and pesticide usage, aiming to analyze their impact on crop yields.
[Source of dataset: Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

## Classification Categories
- **High Yield**: Indicates optimal production conditions and farming practices
- **Medium Yield**: Suggests average production with potential for improvement
- **Low Yield**: Signals need for intervention in farming practices or conditions


### Machine Learning Model Training
To make EzanAi we trained machine learning model where the best performing model was random forest model by using the following steps:

## Training Steps

1. **Preprocessing:** Cleaned and split the data, applied feature engineering and encoding.
2. **Model Optimization:** Used a **pretrained model**, **regularization**, **Adam optimizer**, and **early stopping**.
3. **Hyperparameter Tuning:** Fine-tuned hyperparameters for optimal performance.
4. **Evaluation:** Assessed the model using **accuracy**, **loss**, **precision**, **recall**, and **F1 score**.


## Features
1. **Model Prediction** - Users can input features to get predictions.
2. **Data Visualization** - Meaningful visualizations of key dataset features.
3. **Bulk Data Upload** - Users can upload a CSV file with multiple rows for batch predictions.
4. **Retraining Trigger** - Users can upload new data and trigger model retraining.



## Project Structure
This repository contains the backend application for EzanAi built with python and FastApi.
```
Project_name/
│── README.md
│── notebook/
│   ├──cropyieldclassifier.ipynb     # Jupyter Notebook for model development
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

##  Deployment 
To deploy the EzanAi backend we used render and vercel for frontend

## Frontend repository

Link: https://github.com/lilika67/cropYieldPredictor_fn.git

## How to run this app on frontend
To efficiently use EzanAI app we also created a user friendly web app which will be helping people to test all functionalities.
to test it use [ Frontend link](https://crop-yield-predictor-fn.vercel.app/)



## Related screenshot for EzanAi app

## EzanAi Home Page

![image](https://github.com/user-attachments/assets/8057dfef-1c9c-42ec-b1ba-92fb7cbae535)

## EzanAi prediction form

![image](https://github.com/user-attachments/assets/faa85920-ec2f-4cb0-8f4b-17de2abcd8c7)

## EzanAi visualizations
![image](https://github.com/user-attachments/assets/4b5eb7b0-31c2-41c7-a257-1b4484b103ff)


