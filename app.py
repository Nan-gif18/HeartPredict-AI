import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the model and scaler
@app.route('/')
def home():
    return render_template('index.html')

# Check if model and scaler exist, if not, train them
def load_or_train_model():
    if not os.path.exists('best_model.pkl') or not os.path.exists('scaler.pkl'):
        train_models()
    
    # Load the trained model and scaler
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    
    return model, scaler

def train_models():
    """Train models if they don't exist"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from imblearn.over_sampling import SMOTE
    
    # Load the dataset
    df = pd.read_csv('heart.csv')
    
    # Separate features and target variable
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the feature data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for oversampling the minority class
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)
        score = roc_auc_score(y_test, y_pred)  # Choosing based on ROC AUC Score
        
        # Prefer Random Forest if score is equal
        if score > best_score or (score == best_score and name == 'Random Forest'):
            best_score = score
            best_model = model
    
    # Save the best model and scaler
    pickle.dump(best_model, open('best_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

@app.route('/predict', methods=['POST'])
def predict():
    model, scaler = load_or_train_model()
    
    # Get input features from form
    features = [float(request.form.get(f, 0)) for f in 
                ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    
    # Convert to numpy array and reshape
    final_features = np.array(features).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)
    
    output = "Positive (Heart Disease)" if prediction[0] == 1 else "Negative (Healthy)"
    risk_percentage = round(prediction_proba[0][1] * 100, 2)
    
    return render_template('index.html', 
                           prediction_text=f'Heart Disease Prediction: {output}',
                           risk_text=f'Risk Percentage: {risk_percentage}%',
                           features=dict(zip(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], features)))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    model, scaler = load_or_train_model()
    
    # Get input features from JSON
    data = request.get_json(force=True)
    features = [float(data.get(f, 0)) for f in 
                ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    
    # Convert to numpy array and reshape
    final_features = np.array(features).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)
    
    return jsonify({
        'prediction': int(prediction[0]),
        'risk_percentage': float(prediction_proba[0][1] * 100),
        'message': "Positive (Heart Disease)" if prediction[0] == 1 else "Negative (Healthy)"
    })

@app.route('/about')
def about():
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'The slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
    }
    
    return render_template('about.html', feature_descriptions=feature_descriptions)

if __name__ == '__main__':
    app.run(debug=True)
