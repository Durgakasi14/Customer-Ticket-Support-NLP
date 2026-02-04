<<<<<<< HEAD

from flask import Flask, render_template, request
import joblib
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import math

app = Flask(__name__)
satis_model = joblib.load('satisfaction_model.pkl')
res_model = joblib.load('resolution_model.pkl')
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
import re
def parse_time_to_hours_input(x):
    if x is None or x == '':
        return None
    try:
    
        return float(x)
    except:
        s = str(x).strip().lower()
        m = re.search(r'([0-9]+\.?[0-9])\s(day|days|d)\\b', s)
        if m:
            return float(m.group(1)) * 24.0
        m = re.search(r'([0-9]+\\.?[0-9])\\s(hour|hours|hr|hrs|h)\\b', s)
        if m:
            return float(m.group(1))
        m = re.search(r'([0-9]+\\.?[0-9]*)', s)
        if m:
            return float(m.group(1))
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    age = request.form.get('customer_age', '')
    gender = request.form.get('customer_gender', 'Other')
    product = request.form.get('product_purchased', 'Unknown')
    priority = request.form.get('ticket_priority', 'Unknown')
    t_type = request.form.get('ticket_type', 'Unknown')
    channel = request.form.get('ticket_channel', 'Unknown')
    subject = request.form.get('ticket_subject', '')
    description = request.form.get('ticket_description', '')
    first_resp = request.form.get('first_response_time', '')  # can be '', number or text

    
    try:
        age_f = float(age) if age != '' else None
    except:
        age_f = None

    first_resp_h = parse_time_to_hours_input(first_resp)
    
    row = {
        'Customer Age': age_f if age_f is not None else 0,
        'first_response_hours': first_resp_h if first_resp_h is not None else 0,
        'Ticket Priority': priority,
        'Ticket Type': t_type,
        'Ticket channel': channel,
        'Customer Gender': gender,
        'Product Purchased': product,
        'ticket_text': clean_text(subject + " " + description)
    }

    X = pd.DataFrame([row])
    try:
        sat_pred = satis_model.predict(X)[0]
    except Exception as e:
        sat_pred = f"Error: {e}"
    try:
        res_pred_hours = res_model.predict(X)[0]
         
        if res_pred_hours < 24:
            res_display = f"{res_pred_hours:.1f} hours"
        else:
            days = res_pred_hours / 24.0
            
            res_display = f"{days:.1f} days ({res_pred_hours:.1f} hours)"
    except Exception as e:
        res_display = f"Error: {e}"

    return render_template('result.html', satisfaction=sat_pred, resolution=res_display,
                           subject=subject, description=description)

if __name__ == '__main__':
    app.run(debug=True)
=======

from flask import Flask, render_template, request
import joblib
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import math

app = Flask(__name__)
satis_model = joblib.load('satisfaction_model.pkl')
res_model = joblib.load('resolution_model.pkl')
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
import re
def parse_time_to_hours_input(x):
    if x is None or x == '':
        return None
    try:
    
        return float(x)
    except:
        s = str(x).strip().lower()
        m = re.search(r'([0-9]+\.?[0-9])\s(day|days|d)\\b', s)
        if m:
            return float(m.group(1)) * 24.0
        m = re.search(r'([0-9]+\\.?[0-9])\\s(hour|hours|hr|hrs|h)\\b', s)
        if m:
            return float(m.group(1))
        m = re.search(r'([0-9]+\\.?[0-9]*)', s)
        if m:
            return float(m.group(1))
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    age = request.form.get('customer_age', '')
    gender = request.form.get('customer_gender', 'Other')
    product = request.form.get('product_purchased', 'Unknown')
    priority = request.form.get('ticket_priority', 'Unknown')
    t_type = request.form.get('ticket_type', 'Unknown')
    channel = request.form.get('ticket_channel', 'Unknown')
    subject = request.form.get('ticket_subject', '')
    description = request.form.get('ticket_description', '')
    first_resp = request.form.get('first_response_time', '')  # can be '', number or text

    
    try:
        age_f = float(age) if age != '' else None
    except:
        age_f = None

    first_resp_h = parse_time_to_hours_input(first_resp)
    
    row = {
        'Customer Age': age_f if age_f is not None else 0,
        'first_response_hours': first_resp_h if first_resp_h is not None else 0,
        'Ticket Priority': priority,
        'Ticket Type': t_type,
        'Ticket channel': channel,
        'Customer Gender': gender,
        'Product Purchased': product,
        'ticket_text': clean_text(subject + " " + description)
    }

    X = pd.DataFrame([row])
    try:
        sat_pred = satis_model.predict(X)[0]
    except Exception as e:
        sat_pred = f"Error: {e}"
    try:
        res_pred_hours = res_model.predict(X)[0]
         
        if res_pred_hours < 24:
            res_display = f"{res_pred_hours:.1f} hours"
        else:
            days = res_pred_hours / 24.0
            
            res_display = f"{days:.1f} days ({res_pred_hours:.1f} hours)"
    except Exception as e:
        res_display = f"Error: {e}"

    return render_template('result.html', satisfaction=sat_pred, resolution=res_display,
                           subject=subject, description=description)

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> f2683b71efa83a33bae7236b25186aaef13b731b
