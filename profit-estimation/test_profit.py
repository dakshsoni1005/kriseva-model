import sys
import os
import json

# Ensure project path is accessible
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from calculator.profit import estimate_profit
from api.app import app

def test_calculator_direct():
    print("--- Testing Calculator Module Directly ---")
    report = estimate_profit(crop="Groundnut", yield_val=3.8, farm_area=2)
    print("Report Output:")
    print(json.dumps(report, indent=2))
    
    assert report["crop"] == "Groundnut"
    assert report["yield"] == 3.8
    assert report["farmArea"] == 2
    assert report["production"] == 7.6
    assert report["grossIncome"] == 516800.0
    assert report["cultivationCost"] == 110000.0
    assert report["netProfit"] == 406800.0
    print("[PASS] Calculator module direct test passed successfully!")

def test_api_endpoint():
    print("\n--- Testing API Endpoint via Flask Test Client ---")
    client = app.test_client()
    payload = {
        "crop": "Groundnut",
        "yield": 3.8,
        "farmArea": 2
    }
    response = client.post('/profit', data=json.dumps(payload), content_type='application/json')
    print(f"HTTP Status: {response.status_code}")
    response_json = response.get_json()
    print("API Response Output:")
    print(json.dumps(response_json, indent=2))
    
    assert response.status_code == 200
    assert response_json["netProfit"] == 406800.0
    print("[PASS] API endpoint test passed successfully!")

if __name__ == "__main__":
    test_calculator_direct()
    test_api_endpoint()
