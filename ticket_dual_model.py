import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
def parse_time_to_hours(x):
    """Convert various time formats to hours (float). Return NaN if not parseable."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().lower()
    m = re.search(r'([0-9]+\.?[0-9])\s(day|days|d)\\b', s)
    if m:
        return float(m.group(1)) * 24.0
    m = re.search(r'([0-9]+\.?[0-9])\s(hour|hours|hr|hrs|h)\\b', s)
    if m:
        return float(m.group(1))
    m = re.search(r'([0-9]+\.?[0-9])\s(minute|min|mins)\\b', s)
    if m:
        return float(m.group(1)) / 60.0
    m = re.search(r'pt\s*([0-9]+\.?[0-9]*)h', s)
    if m:
        return float(m.group(1))
    m = re.search(r'pt\s*([0-9]+\.?[0-9]*)m', s)
    if m:
        return float(m.group(1))/60.0
    m = re.search(r'([0-9]+\.?[0-9]*)', s)
    if m:
        return float(m.group(1))
    return np.nan
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df = pd.read_csv('Customersupport.csv')
if 'First Response Time' in df.columns:
    df['first_response_hours'] = df['First Response Time'].apply(parse_time_to_hours)
else:
    df['first_response_hours'] = np.nan

if 'Time to Resolution' in df.columns:
    df['time_to_resolution_hours'] = df['Time to Resolution'].apply(parse_time_to_hours)
else:
    df['time_to_resolution_hours'] = np.nan

df['ticket_text'] = (df.get('Ticket Subject','').fillna('') + " " + df.get('Ticket Description','').fillna('')).apply(clean_text)
structured_numeric = ['Customer Age', 'first_response_hours']
structured_categorical = ['Ticket Priority', 'Ticket Type', 'Ticket channel', 'Customer Gender', 'Product Purchased']
for col in structured_numeric:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df.get(col, 0).fillna(0)

for col in structured_categorical:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = 'Unknown'
satis_col = 'Customer Satisfaction Rating'
df_satis = df.dropna(subset=[satis_col]).copy()
def standardize_satisfaction(val):
    try:
        if isinstance(val, (int, float, np.integer, np.floating)) or str(val).strip().isdigit():
            v = int(float(val))
            mapping = {1:'Worst',2:'Bad',3:'Average',4:'Good',5:'Excellent'}
            return mapping.get(v, str(v))
        else:
            return str(val).strip()
    except Exception:
        return str(val)

df_satis['satisfaction_label'] = df_satis[satis_col].apply(standardize_satisfaction)

X_s = df_satis[structured_numeric + structured_categorical + ['ticket_text']].copy()
y_s = df_satis['satisfaction_label']
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)
num_transform = 'passthrough'  
cat_transform = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
text_transform = TfidfVectorizer(max_features=2000)

preprocessor_s = ColumnTransformer(transformers=[
    ('num', num_transform, structured_numeric),
    ('cat', cat_transform, structured_categorical),
    ('text', text_transform, 'ticket_text')
])

# Pipeline and model (classification)
clf_pipeline = Pipeline([
    ('preproc', preprocessor_s),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

print("Training satisfaction classification model...")
clf_pipeline.fit(Xs_train, ys_train)
ys_pred = clf_pipeline.predict(Xs_test)
print("Classification report for satisfaction:")
print(classification_report(ys_test, ys_pred))

# Save classification model
joblib.dump(clf_pipeline, 'satisfaction_model.pkl')
print("Saved satisfaction_model.pkl")
 # Regression model -> Time to resolution (hours)

df_res = df.dropna(subset=['time_to_resolution_hours']).copy()
df_res = df_res[df_res['time_to_resolution_hours'] >= 0]

# Prepare X/y
X_r = df_res[structured_numeric + structured_categorical + ['ticket_text']].copy()
y_r = df_res['time_to_resolution_hours']

# Split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

# Preprocessor for regression can be same structure
preprocessor_r = ColumnTransformer(transformers=[
    ('num', num_transform, structured_numeric),
    ('cat', cat_transform, structured_categorical),
    ('text', text_transform, 'ticket_text')
])

reg_pipeline = Pipeline([
    ('preproc', preprocessor_r),
    ('reg', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

print("Training resolution time regression model...")
reg_pipeline.fit(Xr_train, yr_train)
yr_pred = reg_pipeline.predict(Xr_test)
mae = mean_absolute_error(yr_test, yr_pred)
rmse =np.sqrt(mean_squared_error(yr_test, yr_pred))
print(f"Regression MAE: {mae:.3f} hours, RMSE: {rmse:.3f} hours")

# Save regression model
joblib.dump(reg_pipeline, 'resolution_model.pkl')
print("Saved resolution_model.pkl")



