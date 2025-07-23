import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

MODEL_FILE = "final_model.pkl"
PIPELINE_FILE = "full_pipeline.pkl"


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
    )
    return full_pipeline


if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    # Load dataset
    housing = pd.read_csv("housing.csv")

    # Create income category for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # Save test set for later evaluation
    strat_test_set.to_csv("input.csv", index=False)

    # Separate features and labels
    housing = strat_train_set.copy()
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    full_pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = full_pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(full_pipeline, PIPELINE_FILE)
    print(f"Model and pipeline saved to {MODEL_FILE} and {PIPELINE_FILE}.")

else:
    model = joblib.load(MODEL_FILE)
    full_pipeline = joblib.load(PIPELINE_FILE)

    # Load test dataset
    input_data = pd.read_csv("input.csv")

    # Separate features and true labels
    if "median_house_value" in input_data.columns:
        y_true = input_data["median_house_value"].copy()
        X_test = input_data.drop("median_house_value", axis=1)
    else:
        raise ValueError(
            "The input.csv file does not contain 'median_house_value' column for comparison."
        )

    # Preprocess and predict
    transformed_input = full_pipeline.transform(X_test)
    predictions = model.predict(transformed_input)

    # Calculate metrics
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    print("Predictions made successfully.")
    print("Sample Predictions vs Actual:")
    for i in range(5):
        print(f"Predicted: {predictions[i]:.2f}, Actual: {y_true.iloc[i]}")

    print("\nModel Performance on Test Data:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R\u00b2:   {r2:.2f} (1.0 = perfect prediction)")

    # Save output with predictions
    output_data = X_test.copy()
    output_data["Actual_median_house_value"] = y_true
    output_data["Predicted_median_house_value"] = predictions
    output_data.to_csv("output.csv", index=False)
    print("Output saved to output.csv.")

    # Scatter Plot: Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, predictions, alpha=0.6, color="blue", edgecolors="k")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Predicted vs Actual Median House Value")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("predicted_vs_actual.png")  # Save as image
    plt.show()
    print("Plot saved as predicted_vs_actual.png.")
