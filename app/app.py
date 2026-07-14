from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('../models/fraud_model.pkl')
defaults = joblib.load('../models/default_values.pkl')
column_order = joblib.load('../models/column_order.pkl')
encoders = joblib.load('../models/encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    amount = float(request.form['amount'])
    product_cd = request.form['product_cd']
    card4 = request.form['card4']
    card6 = request.form['card6']
    email_domain = request.form['email_domain']

    input_data = defaults.copy()

    input_data['TransactionAmt'] = amount
    input_data['ProductCD'] = encoders['ProductCD'].transform([product_cd])[0]
    input_data['card4'] = encoders['card4'].transform([card4])[0]
    input_data['card6'] = encoders['card6'].transform([card6])[0]
    input_data['P_emaildomain'] = encoders['P_emaildomain'].transform([email_domain])[0]

    input_df = pd.DataFrame([input_data])[column_order]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "FRAUD" if prediction == 1 else "LEGIT"
    return render_template('index.html', prediction=result, probability=round(probability*100, 2))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)