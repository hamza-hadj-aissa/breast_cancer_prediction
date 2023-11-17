import joblib
import pandas as pd


def predict_breast_cancer(patient_data):
    # load breast cancer model
    breast_cancer_model = joblib.load(
        "./breast_cancer_prediction_model.joblib")
    # json data to df
    df = pd.DataFrame(patient_data)
    prediction = breast_cancer_model.predict(df)
    print(prediction[0])
    if (prediction[0] == 1):
        return "malignant"
    else:
        return "benign"
