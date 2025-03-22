import joblib
import re
import string
from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import traceback
import nltk
from collections import Counter

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Configure Flask to handle boolean values in JSON
app.config['JSON_SORT_KEYS'] = False
app.json.ensure_ascii = False

# Print current directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Try to load the pre-trained model and vectorizer
try:
    # Check if the model files exist in the current directory
    if os.path.exists('fake_news_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        model_path = 'fake_news_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl'
    # Check if the model files exist in the parent directory
    elif os.path.exists('../fake_news_model.pkl') and os.path.exists('../tfidf_vectorizer.pkl'):
        model_path = '../fake_news_model.pkl'
        vectorizer_path = '../tfidf_vectorizer.pkl'
    # Check in CAPSTONE_PROJECT directory
    elif os.path.exists('../CAPSTONE_PROJECT/fake_news_model.pkl') and os.path.exists('../CAPSTONE_PROJECT/tfidf_vectorizer.pkl'):
        model_path = '../CAPSTONE_PROJECT/fake_news_model.pkl'
        vectorizer_path = '../CAPSTONE_PROJECT/tfidf_vectorizer.pkl'
    else:
        raise FileNotFoundError("Model files not found in expected locations")
    
    print(f"Loading model from {model_path}")
    print(f"Loading vectorizer from {vectorizer_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Successfully loaded pre-trained model and vectorizer!")
    
except Exception as e:
    print(f"Error loading pre-trained model: {e}")
    print("Creating a simple model for demonstration instead...")
    
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
    """Clean and preprocess the input text"""
    try:
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r"\\W", " ", text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        # Return original text if preprocessing fails
        return text

def analyze_text_content(text):
    """Analyze text content for fake news indicators"""
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Basic stats
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_no_stop = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Count word frequencies
        word_freq = Counter(words_no_stop).most_common(15)
        
        # Check for sensationalist language
        sensationalist_words = ['shocking', 'breaking', 'explosive', 'bombshell', 'outrageous', 
                              'unbelievable', 'incredible', 'stunning', 'devastating', 'scandal']
        sensationalist_count = sum(1 for word in words_no_stop if word.lower() in sensationalist_words)
        
        # Check for excessive punctuation in original text
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Check for all caps words (shouting)
        all_caps_count = sum(1 for word in word_tokenize(text) if word.isupper() and len(word) > 3)
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate lexical diversity (unique words / total words)
        lexical_diversity = len(set(words_no_stop)) / len(words_no_stop) if words_no_stop else 0
        
        # Indicators dictionary
        indicators = {
            'statistics': {
                'sentence_count': len(sentences),
                'word_count': len(words),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'lexical_diversity': round(lexical_diversity, 2),
            },
            'stylistic_markers': {
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'all_caps_count': all_caps_count,
                'sensationalist_words_count': sensationalist_count
            },
            'top_words': dict(word_freq),
            'potential_indicators': []
        }
        
        # Generate insights based on indicators
        if exclamation_count > 3:
            indicators['potential_indicators'].append({
                'type': 'warning',
                'text': 'Excessive use of exclamation marks may indicate emotional manipulation'
            })
            
        if all_caps_count > 3:
            indicators['potential_indicators'].append({
                'type': 'warning',
                'text': 'Multiple words in ALL CAPS may indicate sensationalism'
            })
            
        if sensationalist_count > 2:
            indicators['potential_indicators'].append({
                'type': 'warning',
                'text': 'Contains sensationalist language often found in misleading content'
            })
            
        if lexical_diversity < 0.4:
            indicators['potential_indicators'].append({
                'type': 'info',
                'text': 'Low lexical diversity may indicate simplistic language'
            })
            
        if avg_sentence_length > 30:
            indicators['potential_indicators'].append({
                'type': 'info',
                'text': 'Very long sentences may reduce readability and hide misleading information'
            })
            
        return indicators
        
    except Exception as e:
        print(f"Error analyzing text content: {e}")
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
            
            # Get prediction probabilities
            confidence_scores = model.predict_proba(vectorized_text)[0]
            confidence_percentage = float(confidence_scores[prediction] * 100)  # Convert to float
            
            # Calculate additional metrics
            text_length = len(news_text)
            word_count = len(news_text.split())
            
            result = {
                'is_fake': True if prediction == 0 else False,  # Use True/False instead of direct bool
                'is_real': True if prediction == 1 else False,  # Use True/False instead of direct bool
                'confidence': confidence_percentage,
                'prediction': 'Real News' if prediction == 1 else 'Fake News',
                'metrics': {
                    'text_length': text_length,
                    'word_count': word_count,
                    'fake_probability': float(confidence_scores[0] * 100),
                    'real_probability': float(confidence_scores[1] * 100)
                }
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
            
            # Get prediction probabilities
            confidence_scores = model.predict_proba(vectorized_text)[0]
            confidence_percentage = float(confidence_scores[prediction] * 100)
            fake_probability = float(confidence_scores[0] * 100)
            real_probability = float(confidence_scores[1] * 100)
            
            return render_template('result.html', 
                                  prediction='Real News' if prediction == 1 else 'Fake News', 
                                  confidence=confidence_percentage,
                                  fake_probability=fake_probability,
                                  real_probability=real_probability,
                                  text=news_text)
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        if request.is_json:
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
        else:
            return render_template('error.html', error=str(e))

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint for advanced text analysis"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Basic prediction
        processed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = int(model.predict(vectorized_text)[0])
        
        # Get prediction probabilities
        confidence_scores = model.predict_proba(vectorized_text)[0]
        confidence_percentage = float(confidence_scores[prediction] * 100)
        
        # Get detailed text analysis
        analysis_results = analyze_text_content(news_text)
        
        # Create comprehensive result
        result = {
            'is_fake': True if prediction == 0 else False,
            'is_real': True if prediction == 1 else False,
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': confidence_percentage,
            'probabilities': {
                'fake_probability': float(confidence_scores[0] * 100),
                'real_probability': float(confidence_scores[1] * 100)
            },
            'analysis': analysis_results
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in text analysis: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 