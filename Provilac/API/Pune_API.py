# Import necessary libraries
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


# Load the trained LSTM model
#A2_1L
model1 = joblib.load("Pune Random Models/Saved/a2_1l.joblib")
model1_X = joblib.load("Pune Random Models/Scalers/A2 1L/a2_1lx.save")
model1_y = joblib.load("Pune Random Models/Scalers/A2 1L/a2_1ly.save")

#A2_value_1L
model2 = joblib.load("Pune Random Models/Saved/a2_val_1l.joblib")
model2_X = joblib.load("Pune Random Models/Scalers/A2 value 1L/a2_val_1lx.save")
model2_y = joblib.load("Pune Random Models/Scalers/A2 value 1L/a2_val_1ly.save")

#A2_value_500ml
model3 = joblib.load("Pune Random Models/Saved/a2_val_500ml.joblib")
model3_X = joblib.load("Pune Random Models/Scalers/New A2 value 500ml/new_a2_val_500mlx.save")
model3_y = joblib.load("Pune Random Models/Scalers/New A2 value 500ml/new_a2_val_500mly.save")

#Buff_value_1L
model4 = joblib.load("Pune Random Models/Saved/buff_val_1l.joblib")
model4_X = joblib.load("Pune Random Models/Scalers/Buffalo value 1L/buff_val_1lx.save")
model4_y = joblib.load("Pune Random Models/Scalers/Buffalo value 1L/buff_val_1ly.save")

#Buff_value_500ml
model5 = joblib.load("Pune Random Models/Saved/buff_val_500ml.joblib")
model5_X = joblib.load("Pune Random Models/Scalers/Buffalo Value 500ml/buf_val_500mlx.save")
model5_y = joblib.load("Pune Random Models/Scalers/Buffalo Value 500ml/buf_val_500mly.save")

#Buff_1L
model6 = joblib.load("Pune Random Models/Saved/buffl.joblib")
model6_X = joblib.load("Pune Random Models/Scalers/Buff 1L/buff_1lx.save")
model6_y = joblib.load("Pune Random Models/Scalers/Buff 1L/buff_1ly.save")

#Past_1L
model7 = joblib.load("Pune Random Models/Saved/past1l.joblib")
model7_X = joblib.load("Pune Random Models/Scalers/Past 1L/past_1lx.save")
model7_y = joblib.load("Pune Random Models/Scalers/Past 1L/past_1ly.save")

#Raw_value_500ml
model8 = joblib.load("Pune Random Models/Saved/raw_val_0.5l.joblib")
model8_X = joblib.load("Pune Random Models/Scalers/Raw val 0.5l/raw_val_0.5lx.save")
model8_y = joblib.load("Pune Random Models/Scalers/Raw val 0.5l/raw_val_0.5ly.save")

#Raw_1L
model9 = joblib.load("Pune Random Models/Saved/raw1l.joblib")
model9_X = joblib.load("Pune Random Models/Scalers/Raw 1L/raw_1lx.save")
model9_y = joblib.load("Pune Random Models/Scalers/Raw 1L/raw_1ly.save")


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

# Endpoint for time series forecasting
#1
@app.post("/a2_1l")
async def forecast(request: ForecastRequest):
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
@app.post("/a2_val_1l")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model2_X.transform([[year, month, day]])

        # Predict
        output = model2.predict(features)

        # Inverse transform the prediction
        prediction = model2_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#3
@app.post("a2_val_500ml")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model3_X.transform([[year, month, day]])

        # Predict
        output = model3.predict(features)

        # Inverse transform the prediction
        prediction = model3_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#4
@app.post("buff_val_1l")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model4_X.transform([[year, month, day]])

        # Predict
        output = model4.predict(features)

        # Inverse transform the prediction
        prediction = model4_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#5
@app.post("buff_val_500ml")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model5_X.transform([[year, month, day]])

        # Predict
        output = model5.predict(features)

        # Inverse transform the prediction
        prediction = model5_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#6
@app.post("buff_1l")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model6_X.transform([[year, month, day]])

        # Predict
        output = model6.predict(features)

        # Inverse transform the prediction
        prediction = model6_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#7
@app.post("past_1l")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model7_X.transform([[year, month, day]])

        # Predict
        output = model7.predict(features)

        # Inverse transform the prediction
        prediction = model7_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#8
@app.post("raw_val_500ml")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model8_X.transform([[year, month, day]])

        # Predict
        output = model8.predict(features)

        # Inverse transform the prediction
        prediction = model8_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#9
@app.post("raw_1l")
async def forecast(request: ForecastRequest):
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model9_X.transform([[year, month, day]])

        # Predict
        output = model9.predict(features)
        output_keys = list(output.keys())
        prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model9_y.inverse_transform(output.reshape(-1,1))

        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])

        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
