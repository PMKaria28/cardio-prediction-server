# from flask import Flask, render_template, request
# from flask_cors import CORS
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)
# CORS(app)

# # Load the model and the scaler
# model = joblib.load('cardio_model.pkl')
# scaler = joblib.load('scaler.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # 1. Extract raw data from form
#         age = float(request.form['age'])
#         gender = int(request.form['gender'])
#         height = float(request.form['height'])
#         weight = float(request.form['weight'])
#         ap_hi = float(request.form['ap_hi'])
#         ap_lo = float(request.form['ap_lo'])
#         cholesterol = int(request.form['cholesterol'])
#         gluc = int(request.form['gluc'])
#         smoke = int(request.form['smoke'])
#         alco = int(request.form['alco'])
#         active = int(request.form['active'])

#         # 2. Replicate One-Hot Encoding (Objective 2 logic)
#         # Features needed: ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3']
#         chol_2 = 1 if cholesterol == 2 else 0
#         chol_3 = 1 if cholesterol == 3 else 0
#         gluc_2 = 1 if gluc == 2 else 0
#         gluc_3 = 1 if gluc == 3 else 0

#         # Create a DataFrame for processing (to keep track of column names for the scaler)
#         input_data = pd.DataFrame([[
#             age, gender, height, weight, ap_hi, ap_lo, smoke, alco, active, 
#             chol_2, chol_3, gluc_2, gluc_3
#         ]], columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3'])

#         # 3. Apply StandardScaler (Objective 2 logic)
#         numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
#         input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

#         # 4. Make Prediction
#         prediction = model.predict(input_data.values)[0]
        
#         result = "Positive (Risk of Disease)" if prediction == 1 else "Negative (No Disease)"
#         color = "red" if prediction == 1 else "green"

#         return render_template('index.html', prediction_text=result, result_color=color)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS # You'll need to install this: pip install flask-cors
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) # Allows the Next.js frontend to access this API

# Load the model and the scaler
model = joblib.load('cardio_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() # Get JSON data from Next.js
        
        # 1. Extract values
        # Note: Ensure Next.js sends keys matching these names
        age = float(data['age'])
        gender = int(data['gender'])
        height = float(data['height'])
        weight = float(data['weight'])
        ap_hi = float(data['ap_hi'])
        ap_lo = float(data['ap_lo'])
        cholesterol = int(data['cholesterol'])
        gluc = int(data['gluc'])
        smoke = int(data['smoke'])
        alco = int(data['alco'])
        active = int(data['active'])

        # 2. Replicate One-Hot Encoding logic
        chol_2 = 1 if cholesterol == 2 else 0
        chol_3 = 1 if cholesterol == 3 else 0
        gluc_2 = 1 if gluc == 2 else 0
        gluc_3 = 1 if gluc == 3 else 0

        input_df = pd.DataFrame([[
            age, gender, height, weight, ap_hi, ap_lo, smoke, alco, active, 
            chol_2, chol_3, gluc_2, gluc_3
        ]], columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3'])

        # 3. Scaling
        numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # 4. Prediction
        prediction = int(model.predict(input_df.values)[0])
        probability = model.predict_proba(input_df.values)[0][1]

        return jsonify({
            "prediction": prediction,
            "probability": round(float(probability), 2),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(debug=True)