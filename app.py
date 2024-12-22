from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('crop_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f'Recommended Crop: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
