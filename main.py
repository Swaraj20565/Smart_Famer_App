# from fastapi import FastAPI
# import joblib
# import pandas as pd

# app = FastAPI()

# # Load model
# model = joblib.load("crop_model.pkl")

# @app.get("/")
# def home():
#     return {"message": "Crop Recommendation API Running"}

# @app.post("/predict")
# def predict(data: dict):

#     sample = pd.DataFrame([[
#         data["N"],
#         data["P"],
#         data["K"],
#         data["temperature"],
#         data["humidity"],
#         data["ph"],
#         data["rainfall"]
#     ]], columns=["N","P","K","temperature","humidity","ph","rainfall"])

#     prediction = model.predict(sample)

#     return {"recommended_crop": prediction[0]}

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("crop_model.pkl")
# Load models
crop_model = joblib.load("./Fertilizer Reccommedation/models/crop_model.pkl")
fert_model = joblib.load("./Fertilizer Reccommedation/models/fert_model.pkl")

# Load model and encoder
soil_model = joblib.load("./Soil Type Prediction/soil_model.pkl")
lesoil_type = joblib.load("./Soil Type Prediction/soil_type_encoder.pkl")

# Load encoders
le_crop = joblib.load("./Fertilizer Reccommedation/models/crop_encoder.pkl")
le_fert = joblib.load("./Fertilizer Reccommedation/models/fert_encoder.pkl")

@app.get("/")
def home():
    return {"message": "Crop Recommendation API Running"}

@app.post("/predict")
def predict(data: dict):  
    # https://soilhealth.dac.gov.in/slusi-visualisation/

    sample = pd.DataFrame([[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]], columns=["N","P","K","temperature","humidity","ph","rainfall"])

    prediction = model.predict(sample)
    
    return {"crop": prediction[0]}

@app.post("/predicts")
def predict(data: dict):

    sample = pd.DataFrame([[
        data["District_Name"],
        data["Soil_color"],
        data["Nitrogen"],
        data["Phosphorus"],
        data["Potassium"],
        data["pH"],
        data["Rainfall"],
        data["Temperature"]
    ]], columns=[
        "District_Name",
        "Soil_color",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH",
        "Rainfall",
        "Temperature"
    ])

    crop_pred = crop_model.predict(sample)
    fert_pred = fert_model.predict(sample)

    crop_name = le_crop.inverse_transform(crop_pred)[0]
    fert_name = le_fert.inverse_transform(fert_pred)[0]

    return {
        "crop": crop_name,
        "fertilizer": fert_name
    }
  
    
@app.post("/predict-soil")
def predict_soil(
    pH: float,
    EC: float,
    Organic_Carbon: float,
    Nitrogen: int,
    Phosphorus: int,
    Potassium: int,
    Moisture: float,
    Temperature: float,
    Soil_Quality: int
):

    sample = pd.DataFrame([[pH,EC,Organic_Carbon,Nitrogen,Phosphorus,Potassium,Moisture,Temperature,Soil_Quality]],
                          columns=[
                              "pH","EC","Organic_Carbon","Nitrogen","Phosphorus",
                              "Potassium","Moisture","Temperature","Soil_Quality"
                          ])

    soil_pred = soil_model.predict(sample)

    soil_name = str(lesoil_type.classes_[int(soil_pred[0])])

    return {"Predicted Soil": soil_name}