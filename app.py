from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        hours = float(request.form["hours"])
        prediction = model.predict([[hours]])
        return render_template("index.html", prediction=round(prediction[0],2))
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run()