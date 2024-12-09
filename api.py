import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#  Load the saved model
model = joblib.load("model/model.pkl")

# Instantiate FastAPI app
app = FastAPI()


# data Input
class PredictionInput(BaseModel):
    AT: float
    V: float
    AP: float
    RH: float


@app.post("/predict")
def predict(input_data: PredictionInput):
    # prepare input data as a numpy array
    data = np.array([[input_data.AT,
                      input_data.V,
                      input_data.AP,
                      input_data.RH]])

    # make prediction
    prediction = model.predict(data)

    # Return the prediction
    return {"prediction": prediction[0]}


# run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

