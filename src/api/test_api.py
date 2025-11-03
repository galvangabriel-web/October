"""
API Test Script
===============

Quick test to verify API endpoints are working
"""

import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)

    response = requests.get(f"{API_BASE}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)

    response = requests.get(f"{API_BASE}/model/info")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Model Type: {data['model_type']}")
        print(f"Number of Features: {data['num_features']}")
        print("\nTop 5 Features:")
        for feat in data['top_10_features'][:5]:
            print(f"  {feat['rank']}. {feat['name']}: {feat['importance']:.4f}")
        print("✅ Model info retrieved!")
    else:
        print(f"⚠️ Model not loaded: {response.json()}")

def test_feature_extraction():
    """Test feature extraction endpoint"""
    print("\n" + "="*60)
    print("Testing Feature Extraction Endpoint")
    print("="*60)

    # Find a telemetry chunk file
    telemetry_dir = Path("organized_data/barber-motorsports-park/race_unknown/telemetry")
    telemetry_files = list(telemetry_dir.glob("*chunk_001.csv"))

    if not telemetry_files:
        print("⚠️ No telemetry files found - skipping test")
        return

    test_file = telemetry_files[0]
    print(f"Using file: {test_file.name}")

    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'text/csv')}
        response = requests.post(f"{API_BASE}/extract-features", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Laps Extracted: {data['num_laps']}")
        print(f"Features per Lap: {data['num_features']}")
        print(f"First 5 Features: {data['feature_names'][:5]}")
        print("✅ Feature extraction successful!")
    else:
        print(f"❌ Error: {response.json()}")

def test_driver_insights():
    """Test driver insights endpoint"""
    print("\n" + "="*60)
    print("Testing Driver Insights Endpoint")
    print("="*60)

    # Find a telemetry chunk file
    telemetry_dir = Path("organized_data/barber-motorsports-park/race_unknown/telemetry")
    telemetry_files = list(telemetry_dir.glob("*chunk_001.csv"))

    if not telemetry_files:
        print("⚠️ No telemetry files found - skipping test")
        return

    test_file = telemetry_files[0]
    print(f"Using file: {test_file.name}")
    print(f"Analyzing vehicle: #0")

    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'text/csv')}
        response = requests.post(
            f"{API_BASE}/driver-insights?vehicle_number=0",
            files=files
        )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 Performance Summary:")
        for key, value in data['performance_summary'].items():
            print(f"  {key}: {value}")

        print(f"\n💪 Strengths ({len(data['strengths'])}):")
        for s in data['strengths']:
            print(f"  • {s}")

        print(f"\n⚠️ Weaknesses ({len(data['weaknesses'])}):")
        for w in data['weaknesses']:
            print(f"  • {w}")

        print(f"\n💡 Recommendations ({len(data['recommendations'])}):")
        for r in data['recommendations']:
            print(f"  • {r}")

        print("\n✅ Driver insights generated!")
    else:
        print(f"❌ Error: {response.json()}")

def test_prediction():
    """Test lap time prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)

    # Sample features (typical fast lap)
    sample_features = {
        "avg_speed": 150.5,
        "max_speed": 185.2,
        "avg_lateral_g": 1.65,
        "max_lateral_g": 1.95,
        "traction_circle_utilization": 0.85,
        "avg_brake_pressure": 85.3,
        "max_brake_pressure": 145.2,
        "avg_throttle": 62.5,
        "shift_count": 12,
        "speed_variance": 1250.5,
        "cornering_consistency": 0.78
    }

    # Get required features from model info
    model_info_response = requests.get(f"{API_BASE}/model/info")
    if model_info_response.status_code != 200:
        print("⚠️ Model not loaded - skipping prediction test")
        return

    required_features = model_info_response.json()['feature_names']

    # Fill missing features with reasonable defaults
    full_features = {feat: sample_features.get(feat, 0.0) for feat in required_features}

    response = requests.post(
        f"{API_BASE}/predict",
        json={"features": full_features}
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n🎯 Predicted Lap Time: {data['predicted_lap_time']:.2f} seconds")
        print(f"\n🔑 Top Contributing Features:")
        for feat in data['top_features']:
            print(f"  {feat['name']}: {feat['value']:.2f} (importance: {feat['importance']:.4f})")
        print("\n✅ Prediction successful!")
    else:
        print(f"❌ Error: {response.json()}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏁 RACING ANALYTICS API TEST SUITE")
    print("="*60)
    print("\nMake sure the API is running:")
    print("  python -m uvicorn src.api.main:app --reload")
    print("\nThen run this script in another terminal:")
    print("  python src/api/test_api.py")
    print("="*60)

    try:
        # Run tests
        test_health_check()
        test_model_info()
        test_feature_extraction()
        test_driver_insights()
        test_prediction()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")
        print("\nStart it with:")
        print("  python -m uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
