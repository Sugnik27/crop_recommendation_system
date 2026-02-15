🌱 Crop Recommendation System

An end-to-end Machine Learning application that recommends the most suitable crop based on soil nutrients and environmental conditions.
The system is designed as a decision-support tool and provides probabilistic top-K crop recommendations instead of a single hard prediction.

App Link : https://croprecommendationsystem-wjgcye3pzegiew6hugwrpp.streamlit.app/


📌 Problem Statement

Choosing the right crop is a critical decision for farmers and depends on multiple factors such as soil quality and climate conditions.
This project uses machine learning to analyze these parameters and recommend crops scientifically rather than relying on guesswork.


🎯 Features

✔ End-to-end ML pipeline (data → model → deployment)

✔ Multiple ML models trained and evaluated

✔ Best model selected using F1-macro score

✔ Top-3 crop recommendations with probabilities

✔ Confidence score for each prediction

✔ Interactive Streamlit web application

✔ Batch prediction support

✔ Clean, production-ready project structur


🧠 How It Works

1. User provides the following inputs:

-- Nitrogen (N)

-- Phosphorus (P)

-- Potassium (K)

-- Soil pH

-- Temperature

-- Humidity

-- Rainfall

2. The trained ML pipeline:

- Applies preprocessing (imputation & scaling)

- Uses the selected best-performing model

- Predicts probabilities for all crop classes

3. The system returns:

- 🌾 Best recommended crop

- 📊 Confidence score

- 🥈 Top-3 alternative crops

⚠️ Since multiple crops can be suitable under similar conditions, the model intentionally outputs probabilistic recommendations.


🤖 Models Used

The following models were trained and compared:

- Logistic Regression

- Decision Tree

- Random Forest

- K-Nearest Neighbors (KNN)

- Support Vector Machine (SVM)

- Gradient Boosting

Model Selection

-> Evaluation metric: F1-macro

Only the best model is saved and used for deployment

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Example Output

Best Crop: Papaya

Confidence: ~19%

Top-3 Crops:

Papaya

Mungbean

Pomegranate

Lower confidence indicates multiple viable crop options, which reflects real-world agricultural uncertainty.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📦 Batch Prediction

Upload a CSV file containing the same feature columns

The system outputs recommended crops with confidence scores

Predictions can be downloaded as a CSV file

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


🛠️ Tech Stack

Python

NumPy, Pandas

Scikit-learn

XGBoost

Joblib

Streamlit

Git & GitHub

=========================================================================================================================================================================================

🔮 Future Enhancements

Region-aware recommendations

Seasonal crop suggestions

Fertilizer advisory system

Probability calibration

Cloud deployment (Streamlit Cloud / AWS)



👤 Author

Sugnik Mondal
Machine Learning & Data Science Enthusiast
