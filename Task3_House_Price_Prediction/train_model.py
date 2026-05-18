import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("housing.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].mean()
)

encoder = LabelEncoder()

df["ocean_proximity"] = encoder.fit_transform(
    df["ocean_proximity"]
)

X = df.drop("median_house_value", axis=1)

y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

joblib.dump(model, "housing_model.pkl")

print("Model saved successfully")