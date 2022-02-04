import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# load the pickel model
model = pickle.load(open("model.pkl", "rb"))

# definit home page
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]#list comprehension
    features = [np.array(float_features)]
    prediction = model.predict(features) # make prediction
    
    return render_template("index.html", prediction_text = "La prédiction de la qualité du vin est : {}".format(prediction))

    

if __name__ == "__main__":
    app.run(debug=True)
