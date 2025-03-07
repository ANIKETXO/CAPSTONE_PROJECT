import requests
import json

def test_fake_news_api():
    url = "http://127.0.0.1:5000/predict"
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
    
    # Test fake news
    data = {"text": fake_news}
    print(f"Sending request to {url} with data: {data}")
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response text: {response.text}")
    
    try:
        result = response.json()
        print("Fake News Test:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
    
    print("\n")
    
    # Test real news
    data = {"text": real_news}
    print(f"Sending request to {url} with data: {data}")
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response text: {response.text}")
    
    try:
        result = response.json()
        print("Real News Test:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error parsing JSON response: {e}")

if __name__ == "__main__":
    test_fake_news_api() 