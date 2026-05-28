# Import necessary libraries
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained Random Forest model
#A2 Value Pack 1L
model1 = joblib.load("../Saved/Hyderabad/A2 Value Pack 1L.joblib")
model1_X = joblib.load("../Scalers/Hyderabad/A2 Value Pack 1L/A2 Value Pack 1Lx.save")
model1_y = joblib.load("../Scalers/Hyderabad/A2 Value Pack 1L/A2 Value Pack 1Ly.save")


#A2 1L
model2 = joblib.load("../Saved/Hyderabad/A2 1L.joblib")
model2_X = joblib.load("../Scalers/Hyderabad/A2 1L/A2 1Lx.save")
model2_y = joblib.load("../Scalers/Hyderabad/A2 1L/A2 1Ly.save")


#Buffalo Milk 1L
model3 = joblib.load("../Saved/Hyderabad/Buffalo Milk 1L.joblib")
model3_X = joblib.load("../Scalers/Hyderabad/Buffalo Milk 1L/Buffalo Milk 1Lx.save")
model3_y = joblib.load("../Scalers/Hyderabad/Buffalo Milk 1L/Buffalo Milk 1Ly.save")


#New A2 Value Pack 500ml
model4 = joblib.load("../Saved/Hyderabad/New A2 Value Pack 500ml.joblib")
model4_X = joblib.load("../Scalers/Hyderabad/New A2 Value Pack 500ml/New A2 Value Pack 500mlx.save")
model4_y = joblib.load("../Scalers/Hyderabad/New A2 Value Pack 500ml/New A2 Value Pack 500mly.save")


#Buffalo Milk Value Pack 500ml
model5 = joblib.load("../Saved/Hyderabad/Buffalo Milk Value Pack 500ml.joblib")
model5_X = joblib.load("../Scalers/Hyderabad/Buffalo Milk Value Pack 500ml/Buffalo Milk Value Pack 500mlx.save")
model5_y = joblib.load("../Scalers/Hyderabad/Buffalo Milk Value Pack 500ml/Buffalo Milk Value Pack 500mly.save")


#Raw Value Pack 1L
model6 = joblib.load("../Saved/Hyderabad/Raw Value Pack 1L.joblib")
model6_X = joblib.load("../Scalers/Hyderabad/Raw Value Pack 1L/Raw Value Pack 1Lx.save")
model6_y = joblib.load("../Scalers/Hyderabad/Raw Value Pack 1L/Raw Value Pack 1Ly.save")


#Buffalo Value Pack 1L
model7 = joblib.load("../Saved/Hyderabad/Buffalo Value Pack 1L.joblib")
model7_X = joblib.load("../Scalers/Hyderabad/Buffalo Value Pack 1L/Buffalo Value Pack 1Lx.save")
model7_y = joblib.load("../Scalers/Hyderabad/Buffalo Value Pack 1L/Buffalo Value Pack 1Ly.save")


# Create a FastAPI app
app = FastAPI()

# Define a request model for input data
class ForecastRequest(BaseModel):
    year: int
    month: int
    day: int


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# Define a POST endpoint for forecasting
#1
@app.post("/A2_Value_Pack_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model1_X.transform([[year, month, day]])

        # Predict
        output = model1.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model1_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#2
@app.post("/A2_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model2_X.transform([[year, month, day]])

        # Predict
        output = model2.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model2_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#3
@app.post("/Buffalo_Milk_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model3_X.transform([[year, month, day]])

        # Predict
        output = model3.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model3_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#4
@app.post("/New_A2_Value_Pack_500ml")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model4_X.transform([[year, month, day]])

        # Predict
        output = model4.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model4_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#5
@app.post("/Buffalo_Milk_Value_Pack_500ml")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model5_X.transform([[year, month, day]])

        # Predict
        output = model5.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model5_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#6
@app.post("/Raw_Value_Pack_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model6_X.transform([[year, month, day]])

        # Predict
        output = model6.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model6_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#7
@app.post("/Buffalo_Value_Pack_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model7_X.transform([[year, month, day]])

        # Predict
        output = model7.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model7_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8100)