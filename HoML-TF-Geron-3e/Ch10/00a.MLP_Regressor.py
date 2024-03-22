from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, random_state=42
)
print(X_train.shape, X_val.shape, X_test.shape)
mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 50, 50), random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_val_pred = pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"{rmse=}")

y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"{rmse=}")
