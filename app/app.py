from pyexpat import model
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model_path = r"D:/house-price-prediction/app/model/house_price_model.pkl"
with open(model_path, "rb") as f:
    #pickle.dump(model, f)
    model = pickle.load(f)
    print("Model loaded successfully!")

@app.route("/")
def index():
    return render_template("index.html")
          ## only for 2 features ##
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Get form data
#     try:
#         features = [float(request.form[feature]) for feature in request.form]
#         prediction = model.predict([features])[0]
#         return render_template("result.html", prediction=round(prediction, 2))
#     except Exception as e:
#         return f"Error: {e}"

          ##  for multiple features e.g.8  ##
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data and map it to the feature names used in the model
        features = [
            float(request.form["longitude"]),
            float(request.form["latitude"]),
            float(request.form["housing_median_age"]),
            float(request.form["total_rooms"]),
            float(request.form["total_bedrooms"]),
            float(request.form["population"]),
            float(request.form["households"]),
            float(request.form["median_income"])
        ]
        prediction = model.predict([features])[0]
        return render_template("result.html", prediction=round(prediction, 2))
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
