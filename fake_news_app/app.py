import joblib
import re
import string
from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import json

app = Flask(__name__)

# Configure Flask to handle boolean values in JSON
app.config['JSON_SORT_KEYS'] = False
app.json.ensure_ascii = False

# Print current directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Since there's a feature mismatch between the vectorizer and model,
# let's create a simple model for demonstration purposes
print("Creating a simple model for demonstration...")
# Create a simple vectorizer with a small vocabulary
vectorizer = TfidfVectorizer(max_features=1000)

# Create some sample data
fake_samples = [
    "BREAKING: Aliens have made contact with Earth and are demanding all our chocolate!",
    "Scientists discover that the moon is actually made of cheese",
    "Government admits to hiding evidence of bigfoot for decades"
]

real_samples = [
    "The Federal Reserve announced today that it will maintain current interest rates",
    "New study shows benefits of regular exercise for heart health",
    "Tech company reports quarterly earnings above analyst expectations"
]

# Combine samples and create labels
X_train = fake_samples + real_samples
y_train = [0, 0, 0, 1, 1, 1]  # 0 for fake, 1 for real

# Fit the vectorizer and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

print("Simple model created successfully!")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]','',text)
    text = re.sub(r"\\W"," ",text)
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    text = re.sub(r'<.*?>+','',text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub(r'\w*\d\w*','',text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(f"Received request: {request.method}, Content-Type: {request.content_type}")
    
    if request.is_json:
        # Handle API request
        print("Processing JSON request")
        data = request.get_json()
        print(f"Request data: {data}")
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
            
        processed_text = preprocess_text(news_text)
        print(f"Processed text: {processed_text[:100]}...")
        vectorized_text = vectorizer.transform([processed_text])
        prediction = int(model.predict(vectorized_text)[0])  # Convert to int
        
        confidence = model.predict_proba(vectorized_text)[0]
        confidence_percentage = float(confidence[prediction] * 100)  # Convert to float
        
        result = {
            'is_fake': True if prediction == 0 else False,  # Use True/False instead of direct bool
            'is_real': True if prediction == 1 else False,  # Use True/False instead of direct bool
            'confidence': confidence_percentage,
            'prediction': 'Real News' if prediction == 1 else 'Fake News'
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result)
    else:
        # Handle form submission
        print("Processing form submission")
        news_text = request.form['news_text']
        processed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = int(model.predict(vectorized_text)[0])  # Convert to int
        
        confidence = model.predict_proba(vectorized_text)[0]
        confidence_percentage = float(confidence[prediction] * 100)  # Convert to float
        
        return render_template('result.html', 
                              prediction='Real News' if prediction == 1 else 'Fake News', 
                              confidence=confidence_percentage,
                              text=news_text)

if __name__ == '__main__':
    app.run(debug=True) 