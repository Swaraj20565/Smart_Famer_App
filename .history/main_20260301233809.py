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

@app.get("/")
def home():
    return {"message": "Crop Recommendation API Running"}

@app.post("/predict")
def predict(data: dict):

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