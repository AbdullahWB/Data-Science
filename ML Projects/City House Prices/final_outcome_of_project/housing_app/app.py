from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

# Load model and pipeline
model = joblib.load("final_model.pkl")
pipeline = joblib.load("full_pipeline.pkl")

# Load test dataset (for evaluation)
test_data = pd.read_csv("input.csv")
y_true = test_data["median_house_value"]
X_test = test_data.drop("median_house_value", axis=1)

# Preprocess and predict
X_test_prepared = pipeline.transform(X_test)
predictions = model.predict(X_test_prepared)

# Calculate metrics
mse = mean_squared_error(y_true, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, predictions)
r2 = r2_score(y_true, predictions)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        # Collect input values from form
        try:
            longitude = float(request.form["longitude"])
            latitude = float(request.form["latitude"])
            housing_median_age = float(request.form["housing_median_age"])
            total_rooms = float(request.form["total_rooms"])
            total_bedrooms = float(request.form["total_bedrooms"])
            population = float(request.form["population"])
            households = float(request.form["households"])
            median_income = float(request.form["median_income"])
            ocean_proximity = request.form["ocean_proximity"]

            # Create DataFrame for prediction
            input_df = pd.DataFrame(
                [
                    {
                        "longitude": longitude,
                        "latitude": latitude,
                        "housing_median_age": housing_median_age,
                        "total_rooms": total_rooms,
                        "total_bedrooms": total_bedrooms,
                        "population": population,
                        "households": households,
                        "median_income": median_income,
                        "ocean_proximity": ocean_proximity,
                    }
                ]
            )

            input_prepared = pipeline.transform(input_df)
            result = model.predict(input_prepared)[0]
        except:
            result = "Invalid input. Please check your values."

    return render_template(
        "index.html",
        prediction=result,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        r2=round(r2, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)
