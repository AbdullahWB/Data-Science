import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# Load the dataset
housing = pd.read_csv("housing.csv")

# Create a copy of the DataFrame to avoid modifying the original
housing["income_cat"] = pd.cut(
    housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)


# Separate features and labels
housing = strat_train_set.copy()
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

print("Data loaded and preprocessed successfully.")
print(housing, "\n", housing_labels)


num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# Create a pipeline for numerical attributes
num_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("std_scaler", StandardScaler())]
)
# Create a pipeline for categorical attributes
cat_pipeline = Pipeline([("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])

# Combine both pipelines into a ColumnTransformer
full_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
)

# Fit and transform the data using the full pipeline
housing_prepared = full_pipeline.fit_transform(housing)
print("Data preprocessing completed successfully.")
print(housing_prepared.shape, "features after preprocessing.")


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print("Linear Regression RMSE:", lin_rmse)
lin_rmses = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10,
)
print(pd.Series(lin_rmses).describe())


dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10,
)
# print("Decision Tree RMSE:", -tree_rmses.mean())
print(pd.Series(dec_rmses).describe())


random_reg = RandomForestRegressor()
random_reg.fit(housing_prepared, housing_labels)
random_preds = random_reg.predict(housing_prepared)
random_rmse = root_mean_squared_error(housing_labels, random_preds)
# print("Random Forest Regressor RMSE:", random_rmse)
random_rmses = -cross_val_score(
    random_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10,
)
print(pd.Series(random_rmses).describe())
