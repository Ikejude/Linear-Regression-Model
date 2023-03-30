# Import libraries!
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
with open('lr_model', 'rb') as f:
    model = pickle.load(f)
    
@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    rooms = int(request.form['rooms'])
    ocean_prox = int(request.form['ocean_prox'])
    prediction = model.predict([[rooms, ocean_prox]])
    output = round(prediction[0], 2)
    return render_template('index.html', 
                           prediction_text = f"An apartment with {rooms} rooms and located at {ocean_prox} ocean proximity costs ${output}")

if __name__ == "__main__":
    app.run()


