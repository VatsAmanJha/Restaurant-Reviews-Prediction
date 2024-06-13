from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from text_preprocessing import clean_text
import numpy as np

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


class PredictRequest(BaseModel):
    Review: str


class PredictResponse(BaseModel):
    Prediction: int


def prediction(dataframe, column):
    clean_df = clean_text(dataframe, column)
    CORPUS = np.array(clean_df["Review"])
    pred = model.predict(CORPUS)

    result_df = pd.DataFrame({"Review": clean_df[column], "Prediction": pred})

    return result_df


app = FastAPI(title="Restaurant Reviews Prediction")


origins = [
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Convert the request to a DataFrame
    df = pd.DataFrame([request.dict()])

    # Get the prediction
    result_df = prediction(df, "Review")
    prediction_value = int(result_df["Prediction"].iloc[0])

    return PredictResponse(Prediction=prediction_value)
