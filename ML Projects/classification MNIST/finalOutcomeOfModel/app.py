import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable CORS for all routes

# Define paths for your saved models
# Ensure these paths are correct relative to where app.py is run
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_output", "saved_models")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")

# Load the trained models globally when the Flask app starts
# This avoids reloading them for every prediction request, improving performance.
loaded_knn_model = None
loaded_forest_model = None

try:
    loaded_knn_model = joblib.load(KNN_MODEL_PATH)
    print(f"Successfully loaded KNN model from {KNN_MODEL_PATH}")
except FileNotFoundError:
    print(
        f"Error: KNN model file not found at {KNN_MODEL_PATH}. Please ensure it exists."
    )
except Exception as e:
    print(f"Error loading KNN model: {e}")

try:
    loaded_forest_model = joblib.load(RANDOM_FOREST_MODEL_PATH)
    print(f"Successfully loaded Random Forest model from {RANDOM_FOREST_MODEL_PATH}")
except FileNotFoundError:
    print(
        f"Error: Random Forest model file not found at {RANDOM_FOREST_MODEL_PATH}. Please ensure it exists."
    )
except Exception as e:
    print(f"Error loading Random Forest model: {e}")


@app.route("/")
def index():
    """Serves the main HTML drawing pad page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives pixel data from the frontend, makes a prediction using the loaded models,
    and returns the prediction.
    """
    if not loaded_knn_model or not loaded_forest_model:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 500

    try:
        # Get the JSON data from the request body
        data = request.get_json()
        if "pixels" not in data:
            return jsonify({"error": 'Missing "pixels" data in request.'}), 400

        pixels = data["pixels"]

        # Convert the list of pixels to a NumPy array and reshape for the model
        # Models expect a 2D array: (number_of_samples, number_of_features)
        # Here, 1 sample with 784 features.
        input_data = np.array(pixels, dtype=np.uint8).reshape(1, -1)

        # Make predictions using both models
        knn_prediction = int(
            loaded_knn_model.predict(input_data)[0]
        )  # Convert to int for JSON serialization
        forest_prediction = int(loaded_forest_model.predict(input_data)[0])

        return jsonify(
            {
                "knn_prediction": knn_prediction,
                "random_forest_prediction": forest_prediction,
            }
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


if __name__ == "__main__":
    # Run the Flask app
    # debug=True allows for automatic reloading on code changes and provides a debugger
    app.run(debug=True, port=5000)  # You can change the port if 5000 is in use
