# Fake News Detector Web Application

This is a web application that uses machine learning to detect fake news articles. It's built with Flask and uses a pre-trained model to classify news articles as either real or fake.

## Features

- Simple web interface for submitting news articles
- API endpoint for programmatic access
- Real-time classification with confidence scores
- Detailed analysis of the classification result

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure the model files (`fake_news_model.pkl` and `tfidf_vectorizer.pkl`) are in the same directory as `app.py`
4. Run the application:
   ```
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000/`

## API Usage

You can also use the application programmatically via the API:

```python
import requests
import json

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
data = {"text": "Your news article text here"}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(result)
```

The API returns a JSON object with the following fields:
- `is_fake`: Boolean indicating if the article is classified as fake
- `is_real`: Boolean indicating if the article is classified as real
- `confidence`: Confidence percentage of the prediction
- `prediction`: String representation of the prediction ("Real News" or "Fake News")

## Model Information

The model used in this application is a Logistic Regression classifier trained on a dataset of labeled news articles. It uses TF-IDF vectorization to convert text into features that the model can process.

## Limitations

- The model is based on statistical patterns and may not be 100% accurate
- Always verify news from multiple reliable sources
- The model may not perform well on very short texts or texts in languages other than English 