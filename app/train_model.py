import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')

# Data preprocessing
data = data.dropna()  # Remove missing values
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

# Select features and target
# X = data.drop("median_house_value", axis=1)
# y = data["median_house_value"]


# Select features and target
X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]  # Use these 8 features
y = data["median_house_value"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Features (e.g., house characteristics like size, rooms, etc.)
# X_train = np.array([[1200, 3], [1500, 4], [1800, 3], [2400, 5], [3000, 4]])  # Example feature data
# # Labels (house prices corresponding to the features)
# y_train = np.array([250000, 320000, 360000, 500000, 600000])  # Example target data


# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("D:/house-price-prediction/app/model/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved successfully!")


# Check prediction with some sample input
#sample_input = np.array([[1500, 3]])  # New data to predict (e.g., size = 1500 sq. ft, 3 rooms)

sample_input = np.array([[1.5, 2.3, 10, 2000, 500, 300, 100, 3.5]])
predicted_price = model.predict(sample_input)
print("Predicted price for the house:", predicted_price)
