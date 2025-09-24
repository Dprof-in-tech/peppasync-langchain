import os
import logging
import json
import time
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SimpleAnalyticsEngine:
    """LangChain-powered analytics engine for business intelligence"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.database_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': os.getenv('DATABASE_PORT', '5432'),
            'database': os.getenv('DATABASE_NAME', 'peppagenbi'),
            'user': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', '')
        }

    async def execute_analysis(self, analysis_type: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute different types of analytics based on analysis_type"""
        try:
            filters = filters or {}
            
            if analysis_type == "sales_performance":
                return await self._analyze_sales_performance(filters)
            elif analysis_type == "inventory_analysis":
                return await self._analyze_inventory(filters)
            elif analysis_type == "customer_segmentation":
                return await self._analyze_customer_segmentation(filters)
            elif analysis_type == "marketing_performance":
                return await self._analyze_marketing_performance(filters)
            elif analysis_type == "revenue_trends":
                return await self._analyze_revenue_trends(filters)
            elif analysis_type == "product_performance":
                return await self._analyze_product_performance(filters)
            else:
                return {
                    "error": f"Unknown analysis type: {analysis_type}",
                    "available_types": [
                        "sales_performance", "inventory_analysis", "customer_segmentation",
                        "marketing_performance", "revenue_trends", "product_performance"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error in execute_analysis: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "timestamp": int(time.time())
            }

    async def _analyze_sales_performance(self, filters: Dict) -> Dict[str, Any]:
        """Analyze sales performance with AI insights"""
        try:
            # Mock sales data - in production, query actual database
            sales_data = [
                {"month": "2024-01", "revenue": 2500000, "units_sold": 850, "avg_order_value": 2941},
                {"month": "2024-02", "revenue": 2800000, "units_sold": 920, "avg_order_value": 3043},
                {"month": "2024-03", "revenue": 3200000, "units_sold": 1050, "avg_order_value": 3048},
                {"month": "2024-04", "revenue": 2900000, "units_sold": 980, "avg_order_value": 2959},
                {"month": "2024-05", "revenue": 3600000, "units_sold": 1200, "avg_order_value": 3000},
                {"month": "2024-06", "revenue": 4100000, "units_sold": 1350, "avg_order_value": 3037}
            ]
            
            # Calculate metrics
            total_revenue = sum(d["revenue"] for d in sales_data)
            total_units = sum(d["units_sold"] for d in sales_data)
            avg_monthly_revenue = total_revenue / len(sales_data)
            growth_rate = ((sales_data[-1]["revenue"] - sales_data[0]["revenue"]) / sales_data[0]["revenue"]) * 100
            
            # Generate AI insights
            analysis_prompt = f"""
            Sales Performance Analysis for Nigerian Business (NGN):
            
            Monthly Sales Data:
            {json.dumps(sales_data, indent=2)}
            
            Key Metrics:
            - Total Revenue: ₦{total_revenue:,}
            - Total Units Sold: {total_units:,}
            - Average Monthly Revenue: ₦{avg_monthly_revenue:,.0f}
            - Growth Rate (6 months): {growth_rate:.1f}%
            - Latest Month Revenue: ₦{sales_data[-1]['revenue']:,}
            
            Provide comprehensive sales performance analysis including:
            1. Diagnostic insights (why did this happen?)
            2. Predictive insights (what will likely happen?)
            3. Prescriptive recommendations (what actions to take?)
            
            Focus on Nigerian market context and actionable business strategies.
            """
            
            ai_insights = await self._generate_ai_insights(analysis_prompt)
            
            return {
                "analysis_type": "sales_performance",
                "data": sales_data,
                "metrics": {
                    "total_revenue": total_revenue,
                    "total_units_sold": total_units,
                    "avg_monthly_revenue": avg_monthly_revenue,
                    "growth_rate_percent": round(growth_rate, 2),
                    "latest_month_revenue": sales_data[-1]["revenue"],
                    "trend": "increasing" if growth_rate > 0 else "decreasing"
                },
                "ai_insights": ai_insights,
                "timestamp": int(time.time()),
                "currency": "NGN"
            }
            
        except Exception as e:
            logger.error(f"Error in sales performance analysis: {e}")
            return {"error": str(e), "analysis_type": "sales_performance"}

    async def _analyze_inventory(self, filters: Dict) -> Dict[str, Any]:
        """Analyze inventory levels and trends"""
        try:
            # Mock inventory data
            inventory_data = [
                {"product_id": "P001", "product_name": "iPhone 15 Pro", "category": "Electronics", "current_stock": 5, "reorder_level": 20, "unit_cost": 850000, "stock_value": 4250000},
                {"product_id": "P002", "product_name": "Samsung Galaxy S24", "category": "Electronics", "current_stock": 15, "reorder_level": 25, "unit_cost": 720000, "stock_value": 10800000},
                {"product_id": "P003", "product_name": "Nike Air Max", "category": "Fashion", "current_stock": 2, "reorder_level": 15, "unit_cost": 45000, "stock_value": 90000},
                {"product_id": "P004", "product_name": "MacBook Pro 14", "category": "Electronics", "current_stock": 8, "reorder_level": 10, "unit_cost": 1500000, "stock_value": 12000000},
                {"product_id": "P005", "product_name": "Adidas Ultraboost", "category": "Fashion", "current_stock": 25, "reorder_level": 20, "unit_cost": 55000, "stock_value": 1375000}
            ]
            
            # Identify low stock items
            low_stock_items = [item for item in inventory_data if item["current_stock"] <= item["reorder_level"]]
            
            # Calculate metrics
            total_stock_value = sum(item["stock_value"] for item in inventory_data)
            low_stock_value = sum(item["stock_value"] for item in low_stock_items)
            categories = list(set(item["category"] for item in inventory_data))
            
            # Generate AI insights
            analysis_prompt = f"""
            Inventory Analysis for Nigerian Retail Business:
            
            Current Inventory Status:
            {json.dumps(inventory_data, indent=2)}
            
            Low Stock Alerts:
            {json.dumps(low_stock_items, indent=2)}
            
            Key Metrics:
            - Total Products: {len(inventory_data)}
            - Total Stock Value: ₦{total_stock_value:,}
            - Low Stock Items: {len(low_stock_items)}
            - Low Stock Value: ₦{low_stock_value:,}
            - Categories: {categories}
            
            Provide inventory management insights including:
            1. Stock level assessment and risks
            2. Reorder priorities and recommendations
            3. Cash flow impact analysis
            4. Category-specific strategies
            
            Consider Nigerian supply chain challenges and working capital optimization.
            """
            
            ai_insights = await self._generate_ai_insights(analysis_prompt)
            
            return {
                "analysis_type": "inventory_analysis",
                "data": inventory_data,
                "low_stock_alerts": low_stock_items,
                "metrics": {
                    "total_products": len(inventory_data),
                    "total_stock_value": total_stock_value,
                    "low_stock_count": len(low_stock_items),
                    "low_stock_percentage": (len(low_stock_items) / len(inventory_data)) * 100,
                    "categories": categories,
                    "reorder_urgency": "high" if len(low_stock_items) > 2 else "medium"
                },
                "ai_insights": ai_insights,
                "timestamp": int(time.time()),
                "currency": "NGN"
            }
            
        except Exception as e:
            logger.error(f"Error in inventory analysis: {e}")
            return {"error": str(e), "analysis_type": "inventory_analysis"}

    async def _analyze_customer_segmentation(self, filters: Dict) -> Dict[str, Any]:
        """Analyze customer segments and behavior"""
        try:
            # Mock customer segmentation data
            customer_segments = [
                {
                    "segment": "Premium Buyers",
                    "count": 450,
                    "avg_order_value": 85000,
                    "frequency": "Monthly",
                    "total_revenue": 38250000,
                    "characteristics": "High-income professionals, luxury goods preference"
                },
                {
                    "segment": "Regular Customers",
                    "count": 1200,
                    "avg_order_value": 35000,
                    "frequency": "Quarterly",
                    "total_revenue": 42000000,
                    "characteristics": "Middle-income, price-conscious, brand loyal"
                },
                {
                    "segment": "Occasional Buyers",
                    "count": 800,
                    "avg_order_value": 18000,
                    "frequency": "Semi-annual",
                    "total_revenue": 14400000,
                    "characteristics": "Budget-conscious, seasonal purchases, deal-seekers"
                },
                {
                    "segment": "New Customers",
                    "count": 300,
                    "avg_order_value": 25000,
                    "frequency": "First purchase",
                    "total_revenue": 7500000,
                    "characteristics": "Recent acquisition, testing brand, growth potential"
                }
            ]
            
            total_customers = sum(seg["count"] for seg in customer_segments)
            total_revenue = sum(seg["total_revenue"] for seg in customer_segments)
            
            # Generate AI insights
            analysis_prompt = f"""
            Customer Segmentation Analysis for Nigerian Market:
            
            Customer Segments:
            {json.dumps(customer_segments, indent=2)}
            
            Key Metrics:
            - Total Customers: {total_customers:,}
            - Total Revenue: ₦{total_revenue:,}
            - Segments: {len(customer_segments)}
            
            Provide customer segmentation insights including:
            1. Segment performance and growth opportunities
            2. Customer lifetime value analysis
            3. Retention and acquisition strategies by segment
            4. Nigerian market-specific customer behavior patterns
            5. Personalization and marketing recommendations
            
            Consider local purchasing power, cultural preferences, and digital adoption patterns.
            """
            
            ai_insights = await self._generate_ai_insights(analysis_prompt)
            
            return {
                "analysis_type": "customer_segmentation",
                "segments": customer_segments,
                "metrics": {
                    "total_customers": total_customers,
                    "total_revenue": total_revenue,
                    "segment_count": len(customer_segments),
                    "avg_revenue_per_customer": total_revenue / total_customers if total_customers > 0 else 0,
                    "top_segment_by_revenue": max(customer_segments, key=lambda x: x["total_revenue"])["segment"],
                    "top_segment_by_count": max(customer_segments, key=lambda x: x["count"])["segment"]
                },
                "ai_insights": ai_insights,
                "timestamp": int(time.time()),
                "currency": "NGN"
            }
            
        except Exception as e:
            logger.error(f"Error in customer segmentation analysis: {e}")
            return {"error": str(e), "analysis_type": "customer_segmentation"}

    async def _analyze_marketing_performance(self, filters: Dict) -> Dict[str, Any]:
        """Analyze marketing campaign performance"""
        try:
            # Mock marketing performance data
            marketing_data = [
                {"platform": "Facebook", "spend": 450000, "revenue": 1350000, "roas": 3.0, "conversions": 180, "cac": 2500},
                {"platform": "Instagram", "spend": 320000, "revenue": 1280000, "roas": 4.0, "conversions": 160, "cac": 2000},
                {"platform": "Google", "spend": 600000, "revenue": 1800000, "roas": 3.0, "conversions": 200, "cac": 3000},
                {"platform": "TikTok", "spend": 200000, "revenue": 400000, "roas": 2.0, "conversions": 80, "cac": 2500}
            ]
            
            total_spend = sum(d["spend"] for d in marketing_data)
            total_revenue = sum(d["revenue"] for d in marketing_data)
            total_conversions = sum(d["conversions"] for d in marketing_data)
            overall_roas = total_revenue / total_spend if total_spend > 0 else 0
            
            # Generate AI insights
            analysis_prompt = f"""
            Marketing Performance Analysis for Nigerian Digital Campaigns:
            
            Platform Performance:
            {json.dumps(marketing_data, indent=2)}
            
            Key Metrics:
            - Total Ad Spend: ₦{total_spend:,}
            - Total Revenue: ₦{total_revenue:,}
            - Overall ROAS: {overall_roas:.2f}
            - Total Conversions: {total_conversions:,}
            - Average CAC: ₦{sum(d['cac'] for d in marketing_data) / len(marketing_data):,.0f}
            
            Provide marketing performance insights including:
            1. Platform performance comparison and optimization
            2. Budget allocation recommendations
            3. ROAS improvement strategies
            4. Nigerian market platform preferences and trends
            5. Creative and targeting optimization suggestions
            
            Consider local audience behavior and platform adoption in Nigeria.
            """
            
            ai_insights = await self._generate_ai_insights(analysis_prompt)
            
            return {
                "analysis_type": "marketing_performance",
                "data": marketing_data,
                "metrics": {
                    "total_spend": total_spend,
                    "total_revenue": total_revenue,
                    "overall_roas": round(overall_roas, 2),
                    "total_conversions": total_conversions,
                    "best_performing_platform": max(marketing_data, key=lambda x: x["roas"])["platform"],
                    "highest_spend_platform": max(marketing_data, key=lambda x: x["spend"])["platform"],
                    "efficiency_score": "good" if overall_roas > 3.0 else "needs_improvement"
                },
                "ai_insights": ai_insights,
                "timestamp": int(time.time()),
                "currency": "NGN"
            }
            
        except Exception as e:
            logger.error(f"Error in marketing performance analysis: {e}")
            return {"error": str(e), "analysis_type": "marketing_performance"}

    async def _analyze_revenue_trends(self, filters: Dict) -> Dict[str, Any]:
        """Analyze revenue trends and forecasting"""
        try:
            # Mock revenue trend data
            revenue_data = [
                {"period": "2024-Q1", "revenue": 8500000, "growth": 15.2, "units": 2820},
                {"period": "2024-Q2", "revenue": 10600000, "growth": 24.7, "units": 3370},
                {"period": "2024-Q3", "revenue": 11200000, "growth": 5.7, "units": 3550},
                {"period": "2024-Q4", "revenue": 12800000, "growth": 14.3, "units": 3980}
            ]
            
            total_revenue = sum(d["revenue"] for d in revenue_data)
            avg_growth = sum(d["growth"] for d in revenue_data) / len(revenue_data)
            
            # Generate AI insights with forecasting
            analysis_prompt = f"""
            Revenue Trends Analysis for Nigerian Business:
            
            Quarterly Revenue Data:
            {json.dumps(revenue_data, indent=2)}
            
            Key Metrics:
            - Total Annual Revenue: ₦{total_revenue:,}
            - Average Growth Rate: {avg_growth:.1f}%
            - Latest Quarter: ₦{revenue_data[-1]['revenue']:,}
            - Revenue Range: ₦{min(d['revenue'] for d in revenue_data):,} - ₦{max(d['revenue'] for d in revenue_data):,}
            
            Provide revenue trend analysis including:
            1. Growth pattern analysis and seasonality
            2. Revenue forecasting for next 2 quarters
            3. Growth driver identification
            4. Risk factors and mitigation strategies
            5. Nigerian market economic factors impact
            6. Recommendations for sustainable growth
            
            Consider economic conditions, inflation, and market dynamics in Nigeria.
            """
            
            ai_insights = await self._generate_ai_insights(analysis_prompt)
            
            # Simple forecasting (would use more sophisticated models in production)
            next_q_forecast = revenue_data[-1]["revenue"] * (1 + (avg_growth / 100))
            q2_forecast = next_q_forecast * (1 + (avg_growth / 100))
            
            return {
                "analysis_type": "revenue_trends",
                "data": revenue_data,
                "metrics": {
                    "total_revenue": total_revenue,
                    "average_growth_rate": round(avg_growth, 2),
                    "latest_quarter_revenue": revenue_data[-1]["revenue"],
                    "trend_direction": "upward" if avg_growth > 0 else "downward",
                    "growth_consistency": "stable" if max(d["growth"] for d in revenue_data) - min(d["growth"] for d in revenue_data) < 20 else "volatile"
                },
                "forecast": {
                    "next_quarter": round(next_q_forecast),
                    "second_quarter": round(q2_forecast),
                    "confidence": "medium",
                    "assumptions": "Based on historical growth rate average"
                },
                "ai_insights": ai_insights,
                "timestamp": int(time.time()),
                "currency": "NGN"
            }
            
        except Exception as e:
            logger.error(f"Error in revenue trends analysis: {e}")
            return {"error": str(e), "analysis_type": "revenue_trends"}

    async def _analyze_product_performance(self, filters: Dict) -> Dict[str, Any]:
        """Analyze product performance across categories"""
        try:
            # Mock product performance data
            product_data = [
                {"product_id": "P001", "name": "iPhone 15 Pro", "category": "Electronics", "revenue": 12750000, "units": 150, "margin": 25, "rating": 4.8},
                {"product_id": "P002", "name": "Samsung Galaxy S24", "category": "Electronics", "revenue": 10800000, "units": 180, "margin": 22, "rating": 4.6},
                {"product_id": "P003", "name": "Nike Air Max", "category": "Fashion", "revenue": 2250000, "units": 500, "margin": 45, "rating": 4.3},
                {"product_id": "P004", "name": "MacBook Pro 14", "category": "Electronics", "revenue": 18000000, "units": 120, "margin": 20, "rating": 4.9},
                {"product_id": "P005", "name": "Adidas Ultraboost", "category": "Fashion", "revenue": 1925000, "units": 350, "margin": 40, "rating": 4.2}
            ]
            
            total_revenue = sum(d["revenue"] for d in product_data)
            categories = list(set(d["category"] for d in product_data))
            
            # Generate AI insights
            analysis_prompt = f"""
            Product Performance Analysis for Nigerian Retail:
            
            Product Performance Data:
            {json.dumps(product_data, indent=2)}
            
            Key Metrics:
            - Total Products: {len(product_data)}
            - Total Revenue: ₦{total_revenue:,}
            - Categories: {categories}
            - Best Performer: {max(product_data, key=lambda x: x['revenue'])['name']}
            - Highest Margin: {max(product_data, key=lambda x: x['margin'])['name']} ({max(d['margin'] for d in product_data)}%)
            
            Provide product performance analysis including:
            1. Top and underperforming products
            2. Category performance comparison
            3. Margin optimization opportunities
            4. Customer satisfaction correlation
            5. Product mix recommendations
            6. Inventory planning insights
            
            Consider Nigerian consumer preferences and purchasing power.
            """
            
            ai_insights = await self._generate_ai_insights(analysis_prompt)
            
            # Calculate category performance
            category_performance = {}
            for category in categories:
                cat_products = [p for p in product_data if p["category"] == category]
                category_performance[category] = {
                    "total_revenue": sum(p["revenue"] for p in cat_products),
                    "product_count": len(cat_products),
                    "avg_margin": sum(p["margin"] for p in cat_products) / len(cat_products),
                    "avg_rating": sum(p["rating"] for p in cat_products) / len(cat_products)
                }
            
            return {
                "analysis_type": "product_performance",
                "data": product_data,
                "category_performance": category_performance,
                "metrics": {
                    "total_products": len(product_data),
                    "total_revenue": total_revenue,
                    "categories": categories,
                    "best_performer": max(product_data, key=lambda x: x['revenue'])['name'],
                    "highest_margin_product": max(product_data, key=lambda x: x['margin'])['name'],
                    "avg_margin": sum(d["margin"] for d in product_data) / len(product_data),
                    "avg_rating": sum(d["rating"] for d in product_data) / len(product_data)
                },
                "ai_insights": ai_insights,
                "timestamp": int(time.time()),
                "currency": "NGN"
            }
            
        except Exception as e:
            logger.error(f"Error in product performance analysis: {e}")
            return {"error": str(e), "analysis_type": "product_performance"}

    async def _generate_ai_insights(self, prompt: str) -> str:
        """Generate AI insights using OpenAI"""
        try:
            system_message = "You are a business intelligence analyst specializing in the Nigerian market. Provide actionable insights with specific recommendations that consider local market conditions, currency (NGN), and business practices."
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return f"Unable to generate AI insights: {str(e)}"

    def get_available_analysis_types(self) -> List[Dict[str, str]]:
        """Get list of available analysis types"""
        return [
            {"type": "sales_performance", "description": "Analyze sales trends, growth, and performance metrics"},
            {"type": "inventory_analysis", "description": "Monitor stock levels, reorder points, and inventory optimization"},
            {"type": "customer_segmentation", "description": "Segment customers by behavior, value, and characteristics"},
            {"type": "marketing_performance", "description": "Evaluate marketing campaigns, ROAS, and channel effectiveness"},
            {"type": "revenue_trends", "description": "Analyze revenue patterns and generate forecasts"},
            {"type": "product_performance", "description": "Assess product sales, margins, and category performance"}
        ]

    def get_engine_status(self) -> Dict[str, Any]:
        """Get analytics engine status"""
        return {
            "engine": "SimpleAnalyticsEngine",
            "status": "active",
            "llm_model": "gpt-4o-mini",
            "available_analyses": len(self.get_available_analysis_types()),
            "database_connected": True,  # Mock for now
            "last_analysis": None,
            "capabilities": [
                "sales_analysis", "inventory_monitoring", "customer_insights",
                "marketing_optimization", "forecasting", "ai_recommendations"
            ]
        }