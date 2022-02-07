from flask import Flask, url_for, request, jsonify
import joblib

app = Flask(__name__)

#route index
@app.route("/")
def index():
     return "Predict Wine - Hello World!"
 
@app.route("/predict", methods=["POST"])
def spam():
        # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Check mandatory key
        if "input" in req.keys():
            # Load model
            classifier = joblib.load("models/model.joblib")
            # Predict wine quality
            prediction = classifier.predict([req["input"]])
            # Return the result as JSON but first we need to transform the
            # prediction return note of quality of wint
            prediction = str(prediction[0])
            return jsonify({"predict": prediction}), 200
    return jsonify({"msg": "Error: not a JSON or no email key in your request"})

 
if __name__ == "__main__":
    app.run(debug=True)
 