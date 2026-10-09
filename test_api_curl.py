import urllib.request
import json
import traceback

def test_api():
    print("BACKEND URL: http://localhost:5000")
    print("ENDPOINT: /api/predict")
    
    lms = [{"x": 0.5, "y": 0.5, "z": 0.1} for _ in range(21)]
    payload = json.dumps({"hands": [{"label": "Right", "landmarks": lms}]}).encode('utf-8')
    
    req = urllib.request.Request(
        'http://127.0.0.1:5001/api/predict', 
        data=payload, 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"HTTP STATUS: {response.getcode()}")
            print("RESPONSE:")
            print(json.dumps(json.loads(response.read().decode()), indent=2))
    except Exception as e:
        print(f"HTTP STATUS: ERROR")
        print("RESPONSE:")
        if hasattr(e, 'read'):
            print(e.read().decode())
        traceback.print_exc()

if __name__ == "__main__":
    test_api()
