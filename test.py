from flask import Flask, request, jsonify
from flask import render_template
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

df = df.drop_duplicates()

df[(df['Dangerous'] != 'Yes') & (df['Dangerous'] != 'No')]
df_no_null = df.dropna(subset=['Dangerous'])

df1 = df.copy()
# Convert 'symptoms1' to 'symptoms5' to numerical format
le = LabelEncoder()
for col in ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5', 'AnimalName', 'Dangerous']:
    df1[col] = le.fit_transform(df1[col])

X = df1.drop(columns=['Dangerous'])
y = df1['Dangerous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=90)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_svm = SVC(kernel='linear',random_state=0)
model_svm.fit(X_train_scaled, y_train)



@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Corrected line
    data = request.form.to_dict()  # Convert to dictionary for easier access

    features = [float(value) for value in data.values()]
    features_df = pd.DataFrame([features], index=[0])
    y_train_pred = model_svm.predict(X_train_scaled)

    return jsonify({
        "Predicted value : ": float(lrp[0]),
    })


if __name__ == '__main__':
    # print the number of columns
    app.run(debug=True)
