from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

lr = joblib.load("Models/predictor.pickle")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def main():
    if request.method == 'POST':
        PM, ND, CM, SD= float(request.form['PM']), float(request.form['ND']), float(request.form['CM']), float(request.form['SD'])
        lr_pm = lr.predict([[ PM, ND, CM, SD]])

        # print(lr_pm)

    return render_template("index.html", lr_pm = np.round(lr_pm,3))

if __name__ == "__main__":
    app.run(debug = True)
