from fastapi import FastAPI,Request,Form
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse
from typing import Annotated

app = FastAPI(title="Week 5 demo API")

# Load model
model = joblib.load("artifacts/model.joblib")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to my week 6 demo"}

@app.get("/predict/", response_class=HTMLResponse)
def get_predict_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Predictor</title>
        <script>
            
            async function submitForm(event) {
                event.preventDefault(); // STOP the browser's default form submission (which reloads the page)

                const form = document.getElementById('irisForm'); // Get the form element by its ID
                const formData = new FormData(form); // Get all form data as key-value pairs

                // Convert FormData to URL-encoded string to match FastAPI's Form() expectation
                // This ensures the Content-Type will be application/x-www-form-urlencoded
                const urlEncodedData = new URLSearchParams(formData).toString();

                const predictionResultDiv = document.getElementById('predictionResult');
                predictionResultDiv.innerHTML = 'Predicting...'; // Show a loading message

                try {
                    const response = await fetch(form.action, { // Use the form's action attribute (/predict/)
                        method: form.method, // Use the form's method attribute (post)
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded', // Tell the server it's URL-encoded data
                        },
                        body: urlEncodedData, // Send the URL-encoded data
                    });

                    if (response.ok) { // Check if the response was successful (status 200-299)
                        const result = await response.json(); // Parse the JSON response from FastAPI
                        // Update the div with the prediction from the server
                        predictionResultDiv.innerHTML = `Predicted Species: <strong>${result.predicted_class}</strong>`;
                    } else {
                        const errorData = await response.json();
                        predictionResultDiv.innerHTML = `<span style="color: red;">Error: ${errorData.detail || 'Something went wrong'}</span>`;
                        console.error('Server error:', errorData);
                    }
                } catch (error) {
                    predictionResultDiv.innerHTML = `<span style="color: red;">Failed to connect to the server.</span>`;
                    console.error('Network error or other issue:', error);
                }
            }

            // Attach the submitForm function to the form's submit event
            // This ensures the script runs after the DOM is fully loaded
            document.addEventListener('DOMContentLoaded', () => {
                const form = document.getElementById('irisForm');
                form.addEventListener('submit', submitForm);
            });
        </script>
    </head>
    <body>
        <h1>Iris Species Predictor</h1>
        <!-- Added an ID to the form to easily select it in JavaScript -->
        <form id="irisForm" action="/predict/" method="post">
            <label for="sepal_length">Sepal Length:</label><br>
            <input type="number" step="0.01" id="sepal_length" name="sepal_length" required value="5.1"><br><br>

            <label for="sepal_width">Sepal Width:</label><br>
            <input type="number" step="0.01" id="sepal_width" name="sepal_width" required value="3.5"><br><br>

            <label for="petal_length">Petal Length:</label><br>
            <input type="number" step="0.01" id="petal_length" name="petal_length" required value="1.4"><br><br>

            <label for="petal_width">Petal Width:</label><br>
            <input type="number" step="0.01" id="petal_width" name="petal_width" required value="0.2"><br><br>

            <input type="submit" value="Predict Species">
        </form>

        <!-- A div where the prediction result will be displayed -->
        <div id="predictionResult" style="margin-top: 20px; font-weight: bold; color: #333;">
            <!-- Prediction will appear here after submission -->
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict_species(
        sepal_length: Annotated[float, Form()],
        sepal_width: Annotated[float, Form()],
        petal_length: Annotated[float, Form()],
        petal_width: Annotated[float, Form()]):

    
    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return {
        "predicted_class": prediction
    }
# Dummy change to trigger workflow
# Dummy change to trigger workflow
