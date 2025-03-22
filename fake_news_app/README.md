# Fake News Detection Application

An AI-powered web application that analyzes news articles and predicts whether they contain fake or real news using machine learning techniques.

![Fake News Detection](https://img.shields.io/badge/ML-Fake%20News%20Detection-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![TF-IDF](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Python](https://img.shields.io/badge/Language-Python-yellow)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Web Interface](#web-interface)
  - [API Reference](#api-reference)
- [Technical Details](#-technical-details)
- [Advanced Analytics](#-advanced-analytics)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

## ✨ Features

- **News Article Analysis**: Paste any news article text to check its authenticity
- **Predictive Model**: Uses machine learning to identify patterns in fake news
- **Confidence Scores**: Shows prediction confidence with probability breakdown
- **Interactive UI**: User-friendly interface with example texts and loading animations
- **Detailed Reports**: View in-depth analysis of article language and style
- **PDF Export**: Export prediction results as PDF reports
- **RESTful API**: Integrate with other applications via JSON API
- **Advanced Text Analytics**: Get detailed linguistic analysis of content

## 🔧 Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd fake_news_app
   ```

2. **Set up a virtual environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5000/`.

## 🚀 Usage

### Web Interface

1. **Open your browser** and navigate to `http://127.0.0.1:5000/`
2. **Enter or paste** news article text in the provided textarea
3. **Click "Check Authenticity"** to analyze the text
4. **View the results** showing whether the text is likely real or fake news
5. **Export results** to PDF if desired

### API Reference

The application provides a RESTful API for integration with other systems:

#### Basic Prediction Endpoint

**POST `/predict`**

Request:
```json
{
  "text": "Your news article text here..."
}
```

Response:
```json
{
  "is_fake": true,
  "is_real": false,
  "confidence": 92.45,
  "prediction": "Fake News",
  "metrics": {
    "text_length": 253,
    "word_count": 48,
    "fake_probability": 92.45,
    "real_probability": 7.55
  }
}
```

#### Advanced Analytics Endpoint

**POST `/analyze`**

Request:
```json
{
  "text": "Your news article text here..."
}
```

Response:
```json
{
  "is_fake": true,
  "is_real": false,
  "prediction": "Fake News",
  "confidence": 92.45,
  "probabilities": {
    "fake_probability": 92.45,
    "real_probability": 7.55
  },
  "analysis": {
    "statistics": {
      "sentence_count": 5,
      "word_count": 85,
      "avg_sentence_length": 17.0,
      "lexical_diversity": 0.65
    },
    "stylistic_markers": {
      "exclamation_count": 4,
      "question_count": 1,
      "all_caps_count": 3,
      "sensationalist_words_count": 2
    },
    "top_words": {
      "breaking": 3,
      "news": 2,
      "president": 2,
      "...": "..."
    },
    "potential_indicators": [
      {
        "type": "warning",
        "text": "Excessive use of exclamation marks may indicate emotional manipulation"
      },
      {
        "type": "warning",
        "text": "Multiple words in ALL CAPS may indicate sensationalism"
      }
    ]
  }
}
```

## 🔍 Technical Details

### Model Information

- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Training Data**: Combination of real and fake news articles
- **Accuracy**: ~94% on test data

### Text Preprocessing

The application performs several preprocessing steps on input text:
- Converting to lowercase
- Removing special characters and punctuation
- Removing URLs and HTML tags
- Removing numbers and extra whitespace

## 📊 Advanced Analytics

The advanced analytics feature provides deeper insights into text content by analyzing:

1. **Basic Statistics**:
   - Sentence count and word count
   - Average sentence length
   - Lexical diversity (ratio of unique words to total words)

2. **Stylistic Markers**:
   - Use of exclamation marks and question marks
   - ALL CAPS words (often used for emphasis in fake news)
   - Presence of sensationalist language

3. **Potential Indicators**:
   - Auto-generated insights based on textual patterns
   - Warnings about potential manipulation techniques
   - Specific linguistic features that may indicate fake content

## 📁 Project Structure

```
fake_news_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── fake_news_model.pkl    # Trained model
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── test_api.py            # API test script
├── templates/
│   ├── index.html         # Home page template
│   ├── result.html        # Results page template
│   └── error.html         # Error page template
```

## 🔮 Future Improvements

- Implement user accounts to save analysis history
- Add support for analyzing news from URLs directly
- Integrate with fact-checking APIs for additional verification
- Expand the model to support multiple languages
- Build a browser extension for real-time analysis
- Create a mobile application version
- Add more sophisticated linguistic feature extraction
- Implement a confidence threshold for uncertain predictions 