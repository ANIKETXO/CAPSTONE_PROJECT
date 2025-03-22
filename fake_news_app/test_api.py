import requests
import json

def test_fake_news_api():
    base_url = "http://127.0.0.1:5000"
    predict_url = f"{base_url}/predict"
    analyze_url = f"{base_url}/analyze"
    headers = {"Content-Type": "application/json"}
    
    # Test with a fake news example
    fake_news = """
    BREAKING: Aliens have made contact with Earth and are demanding all our chocolate!
    The President has confirmed that extraterrestrial beings arrived at the White House
    last night and their only demand was for Earth to surrender all chocolate supplies.
    Scientists are baffled by this unexpected request. The military has been placed on
    high alert and chocolate factories worldwide have been secured.
    """
    
    # Test with a real news example
    real_news = """
    The Federal Reserve announced today that it will maintain current interest rates,
    citing stable economic indicators and moderate inflation. The decision was widely
    expected by economists and market analysts. In a statement, the Fed Chair noted that
    the labor market remains strong while acknowledging ongoing global economic uncertainties.
    """
    
    print("="*80)
    print("TESTING BASIC PREDICTION API (/predict)")
    print("="*80)
    
    # Test fake news
    data = {"text": fake_news}
    print(f"Sending request to {predict_url} with data: {data}")
    response = requests.post(predict_url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    
    try:
        result = response.json()
        print("Fake News Test Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response.text}")
    
    print("\n")
    
    # Test real news
    data = {"text": real_news}
    print(f"Sending request to {predict_url} with data: {data}")
    response = requests.post(predict_url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    
    try:
        result = response.json()
        print("Real News Test Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response.text}")
    
    print("\n")
    print("="*80)
    print("TESTING ADVANCED ANALYTICS API (/analyze)")
    print("="*80)
    
    # Test advanced analytics with fake news
    data = {"text": fake_news}
    print(f"Sending request to {analyze_url} with fake news data")
    response = requests.post(analyze_url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    
    try:
        result = response.json()
        print("Fake News Advanced Analytics Result:")
        # Only print main structure to keep output manageable
        analytics_summary = {
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "probabilities": result.get("probabilities"),
            "analysis_statistics": result.get("analysis", {}).get("statistics", {}),
            "stylistic_markers": result.get("analysis", {}).get("stylistic_markers", {}),
            "potential_indicators_count": len(result.get("analysis", {}).get("potential_indicators", [])),
        }
        print(json.dumps(analytics_summary, indent=2))
        
        # Print potential indicators separately
        indicators = result.get("analysis", {}).get("potential_indicators", [])
        if indicators:
            print("\nPotential Indicators:")
            for idx, indicator in enumerate(indicators, 1):
                print(f"{idx}. [{indicator.get('type', 'info')}] {indicator.get('text', '')}")
        
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response.text}")
    
    print("\n")
    
    # Test advanced analytics with real news
    data = {"text": real_news}
    print(f"Sending request to {analyze_url} with real news data")
    response = requests.post(analyze_url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    
    try:
        result = response.json()
        print("Real News Advanced Analytics Result:")
        # Only print main structure to keep output manageable
        analytics_summary = {
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "probabilities": result.get("probabilities"),
            "analysis_statistics": result.get("analysis", {}).get("statistics", {}),
            "stylistic_markers": result.get("analysis", {}).get("stylistic_markers", {}),
            "potential_indicators_count": len(result.get("analysis", {}).get("potential_indicators", [])),
        }
        print(json.dumps(analytics_summary, indent=2))
        
        # Print potential indicators separately
        indicators = result.get("analysis", {}).get("potential_indicators", [])
        if indicators:
            print("\nPotential Indicators:")
            for idx, indicator in enumerate(indicators, 1):
                print(f"{idx}. [{indicator.get('type', 'info')}] {indicator.get('text', '')}")
        
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response text: {response.text}")

if __name__ == "__main__":
    test_fake_news_api() 