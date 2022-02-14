from flask import Flask, url_for, request, jsonify, render_template
import joblib

app = Flask(__name__)

#route index
@app.route("/")
def index():
     return render_template("index.html")
 
@app.route('/prediction.', methods=('GET', 'POST'))
def  precition():
    return render_template('prediction.html', prediction)

@app.route("/predict", methods=["POST", "GET"])
def predict():
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
    return jsonify({"msg": "Error: not a JSON or no input key in your request"})

 
if __name__ == "__main__":
    app.run(debug=True)
 