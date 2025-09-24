#!/usr/bin/env python3
"""
Peppasync Lambda Testing Script
Test the new enhanced features locally or on deployed Lambda
"""

import json
import requests
import time
import uuid
from typing import Dict, Any

# Configuration - Update this with your API Gateway URL after deployment
BASE_URL = "http://localhost:8000"  # For local testing
# BASE_URL = "https://your-api-id.execute-api.eu-west-1.amazonaws.com/api"  # For deployed version

def test_enhanced_chat():
    """Test the new enhanced chat with conversation context"""
    print("🔄 Testing Enhanced Chat with Context...")
    
    session_id = str(uuid.uuid4())
    
    # First question
    response1 = requests.post(f"{BASE_URL}/chat", json={
        "prompt": "What were our sales last month?",
        "session_id": session_id
    })
    
    print(f"First Question Response: {response1.status_code}")
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"Response: {result1.get('output', 'No output')[:100]}...")
    
    # Follow-up question (should use context)
    time.sleep(1)
    response2 = requests.post(f"{BASE_URL}/chat", json={
        "prompt": "How does that compare to the previous month?",
        "session_id": session_id
    })
    
    print(f"Follow-up Question Response: {response2.status_code}")
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"Context Used: {result2.get('context_used', False)}")
        print(f"Is Follow-up: {result2.get('is_followup', False)}")
        print(f"Response: {result2.get('output', 'No output')[:100]}...")

def test_analytics_engine():
    """Test the analytics engine"""
    print("\n📊 Testing Analytics Engine...")
    
    analysis_types = ['quick_metrics', 'inventory_status', 'sales_trend']
    
    for analysis_type in analysis_types:
        print(f"  Testing {analysis_type}...")
        response = requests.post(f"{BASE_URL}/analytics/{analysis_type}", json={
            "filters": {}
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✅ {analysis_type}: {len(result.get('data', []))} data points")
            if 'insights' in result:
                print(f"    💡 Insights: {result['insights'][:60]}...")
        else:
            print(f"    ❌ {analysis_type}: Failed ({response.status_code})")

def test_inventory_agent():
    """Test the inventory monitoring agent"""
    print("\n📦 Testing Inventory Agent...")
    
    response = requests.post(f"{BASE_URL}/agents/inventory/run", json={})
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Status: {result.get('status')}")
        print(f"  Products Checked: {result.get('summary', {}).get('products_checked', 0)}")
        print(f"  Critical Alerts: {result.get('summary', {}).get('critical_alerts', 0)}")
        print(f"  Actions Taken: {result.get('summary', {}).get('actions_executed', 0)}")
        
        if result.get('alerts'):
            print(f"  🚨 Sample Alert: {result['alerts'][0].get('product_name', 'Unknown')} - {result['alerts'][0].get('priority', 'Unknown')}")
    else:
        print(f"  ❌ Inventory Agent Failed: {response.status_code}")
        if response.text:
            print(f"  Error: {response.text[:100]}...")

def test_marketing_agent():
    """Test the marketing optimization agent"""
    print("\n📈 Testing Marketing Agent...")
    
    response = requests.post(f"{BASE_URL}/agents/marketing/run", json={})
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Status: {result.get('status')}")
        print(f"  Campaigns Analyzed: {result.get('summary', {}).get('campaigns_analyzed', 0)}")
        print(f"  Opportunities Found: {result.get('summary', {}).get('opportunities_found', 0)}")
        print(f"  Recommendations Generated: {result.get('summary', {}).get('recommendations_generated', 0)}")
        
        if result.get('opportunities'):
            top_opportunity = result['opportunities'][0]
            print(f"  🎯 Top Opportunity: {top_opportunity.get('type', 'Unknown')} - {top_opportunity.get('priority', 'Unknown')}")
    else:
        print(f"  ❌ Marketing Agent Failed: {response.status_code}")
        if response.text:
            print(f"  Error: {response.text[:100]}...")

def test_agent_status():
    """Test the agent status endpoint"""
    print("\n🔍 Testing Agent Status...")
    
    response = requests.get(f"{BASE_URL}/agents/status")
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Inventory Agent: {result.get('inventory_agent', {}).get('status', 'Unknown')}")
        print(f"  Marketing Agent: {result.get('marketing_agent', {}).get('status', 'Unknown')}")
    else:
        print(f"  ❌ Agent Status Failed: {response.status_code}")

def test_backward_compatibility():
    """Test that original endpoints still work"""
    print("\n🔄 Testing Backward Compatibility...")
    
    # Test original retrieve_and_generate
    response = requests.post(f"{BASE_URL}/retrieve_and_generate", json={
        "prompt": "What is our revenue this quarter?"
    })
    
    if response.status_code == 200:
        print("  ✅ Original retrieve_and_generate works")
    else:
        print(f"  ❌ Original retrieve_and_generate failed: {response.status_code}")
    
    # Test original retrieve_and_visualize
    response = requests.post(f"{BASE_URL}/retrieve_and_visualize", json={
        "prompt": "Show me a sales chart"
    })
    
    if response.status_code == 200:
        print("  ✅ Original retrieve_and_visualize works")
    else:
        print(f"  ❌ Original retrieve_and_visualize failed: {response.status_code}")

def main():
    """Run all tests"""
    print("🧪 Starting Peppasync Lambda Feature Tests")
    print(f"📡 Base URL: {BASE_URL}")
    print("=" * 60)
    
    try:
        # Test all new features
        test_enhanced_chat()
        test_analytics_engine()
        test_inventory_agent()
        test_marketing_agent()
        test_agent_status()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("\n💡 To test with your deployed Lambda:")
        print("   1. Update BASE_URL in this script")
        print("   2. Ensure your API Gateway is accessible")
        print("   3. Check AWS CloudWatch logs for detailed error info")
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection failed to {BASE_URL}")
        print("💡 Make sure your server is running:")
        print("   Local: chalice local --host 0.0.0.0 --port 8000")
        print("   AWS: Check your API Gateway URL")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()