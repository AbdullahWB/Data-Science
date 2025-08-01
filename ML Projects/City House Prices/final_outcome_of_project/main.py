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


MODEL_FILE = "final_model.pkl"
PIPELINE_FILE = "full_pipeline.pkl"


def build_pipeline(num_attribs, cat_attribs):
    # Create a pipeline for numerical attributes
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    # Create a pipeline for categorical attributes
    cat_pipeline = Pipeline(
        [("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Combine both pipelines into a ColumnTransformer
    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
    )

    return full_pipeline


if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    # Load the dataset
    housing = pd.read_csv("housing.csv")

    # Create a copy of the DataFrame to avoid modifying the original
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        strat_test_set = (
            housing.loc[test_index]
            .drop("income_cat", axis=1)
            .to_csv("input.csv", index=False)
        )

    # Separate features and labels
    housing = strat_train_set.copy()
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    print("Data loaded and preprocessed successfully.")
    print(housing.head(), "\n", housing_labels.head())

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # Build the full pipeline
    full_pipeline = build_pipeline(num_attribs, cat_attribs)

    # Fit and transform the data using the full pipeline
    housing_prepared = full_pipeline.fit_transform(housing_features)
    print("Data preprocessing completed successfully.")
    print(housing_prepared.shape, "features after preprocessing.")

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    print("Model training completed successfully.")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(full_pipeline, PIPELINE_FILE)
    print(f"Model and pipeline saved to {MODEL_FILE} and {PIPELINE_FILE}.")

else:
    model = joblib.load(MODEL_FILE)
    full_pipeline = joblib.load(PIPELINE_FILE)
    print(f"Model and pipeline loaded from {MODEL_FILE} and {PIPELINE_FILE}.")

    input_data = pd.read_csv("input.csv")
    transformed_input = full_pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    print("Predictions made successfully.")
    print(predictions)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Output saved to output.csv.")
