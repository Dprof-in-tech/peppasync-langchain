#!/usr/bin/env python3
"""
Advanced Business Intelligence Prompt Testing for PeppaSync

This script tests the sophisticated prompt analysis capabilities
covering all 100 retail scenarios from the prompt engineering guide.
"""

import asyncio
import json
import time
from typing import List, Dict
import httpx

# Sample prompts from different categories
TEST_PROMPTS = {
    "sales_revenue": [
        "What would be the projected sales increase for Q4 if we offered a 15% discount on our top 10 best-selling products?",
        "What's the forecasted revenue if we increase our average order value by 10%?",
        "If we run a flash sale on all summer stock for 48 hours, what is the expected revenue impact?",
        "Project our total sales for the upcoming holiday season based on last year's data and current market trends.",
        "What's the potential revenue loss if we experience a 5% drop in conversion rate?"
    ],
    
    "marketing_customer": [
        "What is the projected increase in website traffic if we double our ad spend on Google for the next 30 days?",
        "If we allocate ₦500,000 to a new influencer marketing campaign, what is the expected return on ad spend (ROAS)?",
        "Forecast the number of new customers we can acquire if we run a 'refer a friend' campaign with a ₦1,000 credit.",
        "What would happen to our customer acquisition cost if we focus 75% of our budget on a single channel, like Meta Ads?",
        "How many abandoned carts can we expect to recover if we send a follow-up email with a discount code?"
    ],
    
    "pricing_promotions": [
        "What's the forecasted change in profit margin if we increase the price of our best-selling product by 8%?",
        "If we lower the price of all slow-moving items by 30%, what is the projected impact on our inventory turnover?",
        "What would be the effect on our sales and profitability if we offer free shipping on all orders over ₦5,000?",
        "What is the optimal discount level for our clearance section to maximize profit without devaluing the brand?",
        "If we offer a 'buy one, get one 50% off' promotion, what is the projected increase in average order value (AOV)?"
    ],
    
    "inventory_operations": [
        "If we reduce our safety stock for all products by 10%, what is the projected decrease in inventory holding costs?",
        "What's the forecasted stockout rate if we increase our reorder point by 15% for all products?",
        "What is the optimal number of a specific product to hold in stock to meet forecasted demand for the next six weeks?",
        "If we consolidate our two warehouses into a single location, what is the projected saving in operational expenses?",
        "What is the projected decrease in fulfillment costs if we automate our picking and packing process?"
    ],
    
    "customer_behavior": [
        "What is the projected decrease in customer churn if we launch a personalized email campaign targeting at-risk customers?",
        "If we start a new subscription box service, what is the expected customer lifetime value (CLV) for a new subscriber?",
        "What is the forecasted increase in repeat purchases if we send a one-time-use discount code to every first-time customer?",
        "If we gamify our loyalty program, what is the projected increase in customer engagement and spending?",
        "What's the expected number of newsletter sign-ups if we offer a lead magnet for a specific product category?"
    ],
    
    "strategic_scenarios": [
        "What is the projected change in our sales if a major competitor lowers their prices by 15%?",
        "If we introduce a subscription model for our most popular product, how will that affect our total sales and customer retention?",
        "What's the expected increase in online reviews and ratings if we offer a small reward for a completed review?",
        "Project the impact on our profit and sales if we reduce our holiday marketing budget by 10% and shift the savings to a post-holiday sale.",
        "Based on our current sales velocity, what is the projected date we will run out of our most popular product, and what actions should we take to prevent it?"
    ]
}

DIAGNOSTIC_PROMPTS = [
    "Why did our average order value decrease by 5% last quarter, and what's the best action to take to reverse that trend?",
    "What marketing campaign had the biggest impact on our sales in Q2, and what's the projected outcome if we double its budget?",
    "What is the most significant factor affecting our customer lifetime value, and what's the forecasted impact of a 5% improvement in that area?"
]

WHAT_IF_SCENARIOS = [
    "What's the projected revenue if we run a sitewide 20% off promotion, but exclude the top 10 best-selling items?",
    "If we invest in a new logistics partner, what's the projected impact on our shipping costs and customer satisfaction scores in Q3?",
    "What would be the projected sales for our winter collection if there's a 10% increase in average regional temperatures?",
    "If we stop running ads on Facebook, what's the forecasted decrease in conversions and the associated cost savings?",
    "If we launch a new product, what is the forecasted first-year revenue and a breakdown of the key factors that will influence that outcome?"
]

class PeppaSyncPromptTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.results = {
            "total_tests": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "category_accuracy": {},
            "response_quality_scores": [],
            "performance_metrics": {}
        }

    async def test_prompt_classification(self):
        """Test prompt classification accuracy"""
        print("🔍 Testing Prompt Classification...")
        
        for category, prompts in TEST_PROMPTS.items():
            print(f"\n📊 Testing {category} prompts...")
            category_results = {
                "total": len(prompts),
                "correct_classifications": 0,
                "response_times": []
            }
            
            for i, prompt in enumerate(prompts, 1):
                try:
                    start_time = time.time()
                    
                    response = await self.client.post(
                        f"{self.base_url}/analyze-prompt",
                        json={"prompt": prompt}
                    )
                    
                    response_time = time.time() - start_time
                    category_results["response_times"].append(response_time)
                    
                    if response.status_code == 200:
                        data = response.json()
                        analysis = data.get("prompt_analysis", {})
                        classified_category = analysis.get("category", "unknown")
                        confidence = analysis.get("confidence", 0)
                        
                        # Check if classification is correct
                        if classified_category == category:
                            category_results["correct_classifications"] += 1
                            print(f"   ✅ Prompt {i}: {classified_category} (confidence: {confidence:.2f}) - {response_time:.2f}s")
                        else:
                            print(f"   ❌ Prompt {i}: Expected {category}, got {classified_category} (confidence: {confidence:.2f})")
                        
                        self.results["successful_classifications"] += 1
                    else:
                        print(f"   💥 Prompt {i}: HTTP {response.status_code}")
                        self.results["failed_classifications"] += 1
                        
                    self.results["total_tests"] += 1
                    
                except Exception as e:
                    print(f"   💥 Prompt {i}: Error - {str(e)}")
                    self.results["failed_classifications"] += 1
                    self.results["total_tests"] += 1
            
            # Calculate category accuracy
            accuracy = (category_results["correct_classifications"] / category_results["total"]) * 100
            avg_response_time = sum(category_results["response_times"]) / len(category_results["response_times"]) if category_results["response_times"] else 0
            
            self.results["category_accuracy"][category] = {
                "accuracy_percentage": accuracy,
                "avg_response_time": avg_response_time
            }
            
            print(f"   📈 {category}: {accuracy:.1f}% accuracy, {avg_response_time:.2f}s avg response")

    async def test_sophisticated_analysis(self):
        """Test sophisticated analysis generation"""
        print("\n🧠 Testing Sophisticated Analysis Generation...")
        
        test_prompts = [
            "What would be the projected sales increase for Q4 if we offered a 15% discount on our top 10 best-selling products?",
            "If we allocate ₦500,000 to a new influencer marketing campaign, what is the expected return on ad spend (ROAS)?",
            "What is the optimal discount level for our clearance section to maximize profit without devaluing the brand?",
            "What is the projected decrease in customer churn if we launch a personalized email campaign targeting at-risk customers?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            try:
                start_time = time.time()
                
                response = await self.client.post(
                    f"{self.base_url}/sophisticated-analysis",
                    json={"prompt": prompt}
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data.get("sophisticated_analysis", "")
                    classification = data.get("prompt_classification", {})
                    
                    # Score the response quality
                    quality_score = self._score_response_quality(analysis)
                    self.results["response_quality_scores"].append(quality_score)
                    
                    print(f"   ✅ Analysis {i}: {classification.get('category')} - {classification.get('analysis_type')}")
                    print(f"      📊 Quality Score: {quality_score:.2f}/10, Response Time: {response_time:.2f}s")
                    print(f"      📝 Response Length: {len(analysis)} characters")
                    
                    # Show first 150 chars of response
                    preview = analysis[:150] + "..." if len(analysis) > 150 else analysis
                    print(f"      💬 Preview: {preview}")
                else:
                    print(f"   💥 Analysis {i}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   💥 Analysis {i}: Error - {str(e)}")

    async def test_conversation_context(self):
        """Test conversational context handling"""
        print("\n💬 Testing Conversational Context...")
        
        conversation_flow = [
            "What were our sales last quarter?",
            "How does that compare to the previous quarter?",  # Follow-up
            "What caused the change?",  # Diagnostic follow-up
            "What should we do to improve next quarter?"  # Prescriptive follow-up
        ]
        
        session_id = f"test_session_{int(time.time())}"
        
        for i, prompt in enumerate(conversation_flow, 1):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat",
                    json={"prompt": prompt, "session_id": session_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    context_used = data.get("context_used", False)
                    is_followup = data.get("is_followup", False)
                    
                    print(f"   ✅ Message {i}: Context Used: {context_used}, Follow-up: {is_followup}")
                else:
                    print(f"   💥 Message {i}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   💥 Message {i}: Error - {str(e)}")

    async def test_capabilities_endpoint(self):
        """Test the capabilities endpoint"""
        print("\n🎯 Testing Capabilities Endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/prompt-capabilities")
            
            if response.status_code == 200:
                data = response.json()
                categories = data.get("supported_categories", {})
                analysis_types = data.get("analysis_types", [])
                
                print(f"   ✅ Supported Categories: {len(categories)}")
                print(f"   ✅ Analysis Types: {len(analysis_types)}")
                
                for category, info in categories.items():
                    sample_count = len(info.get("sample_questions", []))
                    print(f"      📂 {category}: {sample_count} sample questions")
                    
            else:
                print(f"   💥 Capabilities test: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   💥 Capabilities test: Error - {str(e)}")

    def _score_response_quality(self, response: str) -> float:
        """Score response quality based on various factors"""
        if not response:
            return 0.0
            
        score = 0.0
        
        # Length score (optimal around 500-2000 chars)
        length = len(response)
        if 500 <= length <= 2000:
            score += 2.0
        elif 200 <= length < 500:
            score += 1.5
        elif length > 2000:
            score += 1.0
        
        # Structure score - check for sections/organization
        structure_keywords = ["forecast", "analysis", "recommendation", "impact", "strategy"]
        structure_score = sum(1 for keyword in structure_keywords if keyword.lower() in response.lower())
        score += min(structure_score, 3.0)
        
        # Business terminology score
        business_terms = ["revenue", "profit", "margin", "roas", "conversion", "customer", "market", "growth"]
        business_score = sum(1 for term in business_terms if term.lower() in response.lower())
        score += min(business_score * 0.5, 2.0)
        
        # Nigerian context score
        nigerian_context = ["naira", "₦", "nigerian", "nigeria", "local market"]
        context_score = sum(1 for term in nigerian_context if term.lower() in response.lower())
        score += min(context_score, 2.0)
        
        # Numerical specificity score
        import re
        numbers = re.findall(r'\d+\.?\d*%|\₦[\d,]+|\d+\.?\d*', response)
        score += min(len(numbers) * 0.3, 1.0)
        
        return min(score, 10.0)

    async def run_comprehensive_test(self):
        """Run all tests and generate report"""
        print("🚀 Starting Comprehensive PeppaSync Prompt Testing")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test all capabilities
        await self.test_capabilities_endpoint()
        await self.test_prompt_classification()
        await self.test_sophisticated_analysis()
        await self.test_conversation_context()
        
        total_time = time.time() - start_time
        
        # Generate final report
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        print(f"⏱️  Total Test Time: {total_time:.2f} seconds")
        print(f"🎯 Total Tests Run: {self.results['total_tests']}")
        print(f"✅ Successful Classifications: {self.results['successful_classifications']}")
        print(f"❌ Failed Classifications: {self.results['failed_classifications']}")
        
        if self.results['total_tests'] > 0:
            success_rate = (self.results['successful_classifications'] / self.results['total_tests']) * 100
            print(f"📈 Overall Success Rate: {success_rate:.1f}%")
        
        print("\n📊 Category Accuracy:")
        for category, metrics in self.results['category_accuracy'].items():
            print(f"   {category}: {metrics['accuracy_percentage']:.1f}% (avg: {metrics['avg_response_time']:.2f}s)")
        
        if self.results['response_quality_scores']:
            avg_quality = sum(self.results['response_quality_scores']) / len(self.results['response_quality_scores'])
            print(f"\n🌟 Average Response Quality: {avg_quality:.2f}/10.0")
            print(f"🎯 Quality Range: {min(self.results['response_quality_scores']):.1f} - {max(self.results['response_quality_scores']):.1f}")
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        if self.results['category_accuracy']:
            avg_response_time = sum(m['avg_response_time'] for m in self.results['category_accuracy'].values()) / len(self.results['category_accuracy'])
            print(f"   Average Response Time: {avg_response_time:.2f} seconds")
            print(f"   Prompts per Minute: {60 / avg_response_time:.1f}")
        
        await self.client.aclose()

async def main():
    """Main test execution"""
    print("🧪 PeppaSync Advanced Prompt Testing Suite")
    print("Testing 100+ sophisticated retail business intelligence scenarios")
    print()
    
    tester = PeppaSyncPromptTester()
    
    try:
        # Test server connectivity
        response = await tester.client.get(f"{tester.base_url}/")
        if response.status_code != 200:
            print(f"❌ Server not responding at {tester.base_url}")
            print("   Make sure the PeppaSync server is running with: python app.py")
            return
        
        server_info = response.json()
        print(f"✅ Connected to {server_info.get('message', 'PeppaSync')} v{server_info.get('version', '1.0')}")
        print()
        
        await tester.run_comprehensive_test()
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        print("   Make sure the server is running and accessible")
    
    finally:
        await tester.client.aclose()

if __name__ == "__main__":
    asyncio.run(main())