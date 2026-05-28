# Import necessary libraries
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained Random Forest model
#A2 1L
model1 = joblib.load("../Saved/Mumbai/A2 1L.joblib")
model1_X = joblib.load("../Scalers/Mumbai/A2 1L/A2 1Lx.save")
model1_y = joblib.load("../Scalers/Mumbai/A2 1L/A2 1Ly.save")


#A2 Value Pack 1L
model2 = joblib.load("../Saved/Mumbai/A2 Value Pack 1L.joblib")
model2_X = joblib.load("../Scalers/Mumbai/A2 Value Pack 1L/A2 Value Pack 1Lx.save")
model2_y = joblib.load("../Scalers/Mumbai/A2 Value Pack 1L/A2 Value Pack 1Ly.save")


#Buffalo Milk 1L
model3 = joblib.load("../Saved/Mumbai/Buffalo Milk 1L.joblib")
model3_X = joblib.load("../Scalers/Mumbai/Buffalo Milk 1L/Buffalo Milk 1Lx.save")
model3_y = joblib.load("../Scalers/Mumbai/Buffalo Milk 1L/Buffalo Milk 1Ly.save")


#Paneer 200g Cows Milk
model4 = joblib.load("../Saved/Mumbai/Paneer 200g Cows Milk.joblib")
model4_X = joblib.load("../Scalers/Mumbai/Paneer 200g Cows Milk/Paneer 200g Cows Milkx.save")
model4_y = joblib.load("../Scalers/Mumbai/Paneer 200g Cows Milk/Paneer 200g Cows Milky.save")


#Past Milk 1L
model5 = joblib.load("../Saved/Mumbai/Past Milk 1L.joblib")
model5_X = joblib.load("../Scalers/Mumbai/Past Milk 1L/Past Milk 1Lx.save")
model5_y = joblib.load("../Scalers/Mumbai/Past Milk 1L/Past Milk 1Ly.save")


#Raw Milk 1L
model6 = joblib.load("../Saved/Mumbai/Raw Milk 1L.joblib")
model6_X = joblib.load("../Scalers/Mumbai/Raw Milk 1L/Raw Milk 1Lx.save")
model6_y = joblib.load("../Scalers/Mumbai/Raw Milk 1L/Raw Milk 1Ly.save")


#New A2 Value Pack 500ml
model7 = joblib.load("../Saved/Mumbai/New A2 Value Pack 500ml.joblib")
model7_X = joblib.load("../Scalers/Mumbai/New A2 Value Pack 500ml/New A2 Value Pack 500mlx.save")
model7_y = joblib.load("../Scalers/Mumbai/New A2 Value Pack 500ml/New A2 Value Pack 500mly.save")


#Past Milk 0.5L
model8 = joblib.load("../Saved/Mumbai/Past Milk 0.5L.joblib")
model8_X = joblib.load("../Scalers/Mumbai/Past Milk 0.5L/Past Milk 0.5Lx.save")
model8_y = joblib.load("../Scalers/Mumbai/Past Milk 0.5L/Past Milk 0.5Ly.save")


#Buffalo Milk Value Pack 500ml
model9 = joblib.load("../Saved/Mumbai/Buffalo Milk Value Pack 500ml.joblib")
model9_X = joblib.load("../Scalers/Mumbai/Buffalo Milk Value Pack 500ml/Buffalo Milk Value Pack 500mlx.save")
model9_y = joblib.load("../Scalers/Mumbai/Buffalo Milk Value Pack 500ml/Buffalo Milk Value Pack 500mly.save")


#Past Milk Value Pack 1L
model10 = joblib.load("../Saved/Mumbai/Past Milk Value Pack 1L.joblib")
model10_X = joblib.load("../Scalers/Mumbai/Past Milk Value Pack 1L/Past Milk Value Pack 1Lx.save")
model10_y = joblib.load("../Scalers/Mumbai/Past Milk Value Pack 1L/Past Milk Value Pack 1Ly.save")


#Raw Value Pack 1L
model11 = joblib.load("../Saved/Mumbai/Raw Value Pack 1L.joblib")
model11_X = joblib.load("../Scalers/Mumbai/Raw Value Pack 1L/Raw Value Pack 1Lx.save")
model11_y = joblib.load("../Scalers/Mumbai/Raw Value Pack 1L/Raw Value Pack 1Ly.save")


#Lactose Free Milk 0.5L
model12 = joblib.load("../Saved/Mumbai/Lactose Free Milk 0.5L.joblib")
model12_X = joblib.load("../Scalers/Mumbai/Lactose Free Milk 0.5L/Lactose Free Milk 0.5Lx.save")
model12_y = joblib.load("../Scalers/Mumbai/Lactose Free Milk 0.5L/Lactose Free Milk 0.5Ly.save")


#Buffalo Value Pack 1L
model13 = joblib.load("../Saved/Mumbai/Buffalo Value Pack 1L.joblib")
model13_X = joblib.load("../Scalers/Mumbai/Buffalo Value Pack 1L/Buffalo Value Pack 1Lx.save")
model13_y = joblib.load("../Scalers/Mumbai/Buffalo Value Pack 1L/Buffalo Value Pack 1Ly.save")


#Past Milk Value Pack 0.5L
model14 = joblib.load("../Saved/Mumbai/Past Milk Value Pack 0.5L.joblib")
model14_X = joblib.load("../Scalers/Mumbai/Past Milk Value Pack 0.5L/Past Milk Value Pack 0.5Lx.save")
model14_y = joblib.load("../Scalers/Mumbai/Past Milk Value Pack 0.5L/Past Milk Value Pack 0.5Ly.save")


#Raw Milk Value Pack 0.5L
model15 = joblib.load("../Saved/Mumbai/Raw Milk Value Pack 0.5L.joblib")
model15_X = joblib.load("../Scalers/Mumbai/Raw Milk Value Pack 0.5L/Raw Milk Value Pack 0.5Lx.save")
model15_y = joblib.load("../Scalers/Mumbai/Raw Milk Value Pack 0.5L/Raw Milk Value Pack 0.5Ly.save")


#Homogenised A2 1L
model16 = joblib.load("../Saved/Mumbai/Homogenised A2 1L.joblib")
model16_X = joblib.load("../Scalers/Mumbai/Homogenised A2 1L/Homogenised A2 1Lx.save")
model16_y = joblib.load("../Scalers/Mumbai/Homogenised A2 1L/Homogenised A2 1Ly.save")


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
@app.post("/A2_1L")
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
@app.post("/A2_Value_Pack_1L")
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
@app.post("/Paneer_200g_Cows_Milk")
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
@app.post("/Past_Milk_1L")
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
@app.post("/Raw_Milk_1L")
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
@app.post("/New_A2_Value_Pack_500ml")
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



#8
@app.post("/Past_Milk_0.5L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model8_X.transform([[year, month, day]])

        # Predict
        output = model8.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model8_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#9
@app.post("/Buffalo_Milk_Value_Pack_500ml")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model9_X.transform([[year, month, day]])

        # Predict
        output = model9.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model9_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#10
@app.post("/Past_Milk_Value_Pack_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model10_X.transform([[year, month, day]])

        # Predict
        output = model10.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model10_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#11
@app.post("/Raw_Value_Pack_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model11_X.transform([[year, month, day]])

        # Predict
        output = model11.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model11_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#12
@app.post("/Lactose_Free_Milk_0.5L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model12_X.transform([[year, month, day]])

        # Predict
        output = model12.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model12_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#13
@app.post("/Buffalo_Value_Pack_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model13_X.transform([[year, month, day]])

        # Predict
        output = model13.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model13_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#14
@app.post("/Past_Milk_Value_Pack_0.5L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model14_X.transform([[year, month, day]])

        # Predict
        output = model14.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model14_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#15
@app.post("/Raw_Milk_Value_Pack_0.5L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model15_X.transform([[year, month, day]])

        # Predict
        output = model15.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model15_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#16
@app.post("/Homogenised_A2_1L")
def forecast(request: ForecastRequest): 
    try:
        # Extract date input
        year = request.year
        month = request.month
        day = request.day

        # Transform features using the loaded scalers
        features = model16_X.transform([[year, month, day]])

        # Predict
        output = model16.predict(features)
        # output_keys = list(output.keys())
        # prediction_scaled = output[output_keys[0]].numpy()

        # Inverse transform the prediction
        prediction = model16_y.inverse_transform(output.reshape(-1, 1))
        # Convert the prediction to a regular float
        forecasted_value = float(prediction[0][0])
        return {"forecasted_value": forecasted_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8100)