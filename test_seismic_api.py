#!/usr/bin/env python3
"""
Test script for Seismic Detection Agent API endpoints.

Prueba todos los endpoints del agente de detección sísmica.
"""

import requests
import json
import time
from datetime import datetime

def test_seismic_scan():
    """Test manual seismic scan endpoint."""
    print("🔍 Testing seismic scan endpoint...")

    url = "http://localhost:8000/seismic/scan"
    payload = {
        "coordinates": {"lat": 19.4326, "lon": -99.1332},  # Ciudad de México
        "zone_name": "test_zone"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Seismic scan successful!")
            print(f"   Probability: {result.get('seismic_probability', 'N/A')}/10.0")
            print(f"   Intensity: {result.get('intensity', 'N/A')}")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            return True
        else:
            print(f"❌ Scan failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_add_monitoring_zone():
    """Test add monitoring zone endpoint."""
    print("\n📍 Testing add monitoring zone endpoint...")

    url = "http://localhost:8000/seismic/zones/add"
    payload = {
        "name": "Test_Zone_API",
        "coordinates": {"lat": 35.6895, "lon": 139.6917},  # Tokio
        "radius_km": 100.0,
        "priority": "high"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Zone added successfully!")
            print(f"   Zone: {result['zone']['name']}")
            return True
        else:
            print(f"❌ Add zone failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_get_zones():
    """Test get monitoring zones endpoint."""
    print("\n📋 Testing get monitoring zones endpoint...")

    url = "http://localhost:8000/seismic/zones"

    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Zones retrieved successfully!")
            print(f"   Total zones: {result['count']}")
            for zone in result['zones']:
                print(f"   - {zone['name']}: {zone['coordinates']} (radius: {zone['radius_km']}km)")
            return True
        else:
            print(f"❌ Get zones failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_get_stats():
    """Test get seismic detector stats endpoint."""
    print("\n📊 Testing get stats endpoint...")

    url = "http://localhost:8000/seismic/stats"

    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            stats = result['stats']
            print("✅ Stats retrieved successfully!")
            print(f"   Total scans: {stats.get('total_scans', 'N/A')}")
            print(f"   Alerts generated: {stats.get('alerts_generated', 'N/A')}")
            print(f"   Monitoring zones: {stats.get('monitoring_zones', 'N/A')}")
            return True
        else:
            print(f"❌ Get stats failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_start_monitoring():
    """Test start monitoring endpoint."""
    print("\n🚀 Testing start monitoring endpoint...")

    url = "http://localhost:8000/seismic/monitoring/start"

    try:
        response = requests.post(url, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("✅ Monitoring started successfully!")
                print(f"   Active zones: {result.get('active_zones', 'N/A')}")
                return True
            else:
                print(f"⚠️  Monitoring already active: {result['message']}")
                return True
        else:
            print(f"❌ Start monitoring failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_get_alerts_history():
    """Test get alerts history endpoint."""
    print("\n📚 Testing get alerts history endpoint...")

    url = "http://localhost:8000/seismic/alerts/history?limit=5"

    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Alerts history retrieved successfully!")
            print(f"   Total alerts in history: {result['count']}")
            if result['alerts']:
                print("   Recent alerts:")
                for alert in result['alerts'][-3:]:  # Show last 3
                    prob = alert.get('seismic_probability', 'N/A')
                    coords = alert.get('coordinates', {})
                    print(f"     - {alert['timestamp'][:19]}: Prob {prob}/10.0 at {coords}")
            return True
        else:
            print(f"❌ Get alerts history failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("🔔 Seismic Detection Agent API Test Suite")
    print("=" * 50)
    print("Testing all endpoints of the seismic detection agent...")
    print()

    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not running or not healthy")
            print("Please start the server with: python simple_api_server.py")
            return
    except:
        print("❌ Cannot connect to API server")
        print("Please start the server with: python simple_api_server.py")
        return

    print("✅ API server is running")

    # Run tests
    tests = [
        test_seismic_scan,
        test_add_monitoring_zone,
        test_get_zones,
        test_get_stats,
        test_start_monitoring,
        test_get_alerts_history,
    ]

    results = []
    for test in tests:
        results.append(test())
        time.sleep(0.5)  # Small delay between tests

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Seismic Detection Agent API is working correctly.")
        print("\n📋 Available endpoints:")
        print("  POST /seismic/scan              - Manual seismic scan")
        print("  POST /seismic/zones/add         - Add monitoring zone")
        print("  GET  /seismic/zones             - List monitoring zones")
        print("  GET  /seismic/stats             - Detector statistics")
        print("  POST /seismic/monitoring/start  - Start real-time monitoring")
        print("  GET  /seismic/alerts/history    - Alerts history")
    else:
        print("\n💥 SOME TESTS FAILED. Check the implementation.")

if __name__ == "__main__":
    main()