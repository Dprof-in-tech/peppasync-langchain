"""
Expert Marketing & Sales Knowledge Base
Pre-curated insights from top industry voices + dynamic web content
"""
import logging
import asyncio
from typing import List, Dict
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class ExpertKnowledgeBase:
    """Curated marketing and sales knowledge from industry experts"""
    
    @staticmethod
    def get_marketing_insights() -> List[Document]:
        """Marketing strategies from top voices like Gary Vaynerchuk, Neil Patel, etc."""
        
        marketing_insights = [
            {
                "content": "Focus on providing value before asking for anything in return. The jab, jab, jab, right hook method - give value 3 times before making any sales pitch. Build relationships first, sales will follow.",
                "source": "Gary Vaynerchuk",
                "category": "relationship_marketing",
                "topic": "value_first_approach"
            },
            {
                "content": "Content is king, but context is God. Create content that matches where your audience is in their buyer journey. Awareness stage needs educational content, consideration stage needs comparison content, decision stage needs social proof.",
                "source": "Gary Vaynerchuk",
                "category": "content_marketing", 
                "topic": "buyer_journey_content"
            },
            {
                "content": "The best marketing doesn't feel like marketing. Focus on storytelling and emotional connection rather than direct selling. People buy with emotion and justify with logic.",
                "source": "Seth Godin",
                "category": "emotional_marketing",
                "topic": "storytelling"
            },
            {
                "content": "Retargeting campaigns typically have 10x higher CTR than regular display ads. Target users who visited product pages but didn't purchase with specific product ads and limited-time offers.",
                "source": "Neil Patel",
                "category": "paid_advertising",
                "topic": "retargeting"
            },
            {
                "content": "Email marketing has an average ROI of $42 for every $1 spent. Segment your email lists by behavior, demographics, and purchase history for 3x better engagement rates.",
                "source": "HubSpot Research",
                "category": "email_marketing",
                "topic": "segmentation"
            },
            {
                "content": "Social proof increases conversions by up to 15%. Use customer reviews, testimonials, user-generated content, and trust badges prominently on product pages and checkout.",
                "source": "ConversionXL",
                "category": "conversion_optimization",
                "topic": "social_proof"
            },
            {
                "content": "Video content generates 1200% more shares than text and image content combined. Short-form videos (15-60 seconds) perform best on social platforms for product demos and testimonials.",
                "source": "Wistia Research",
                "category": "video_marketing",
                "topic": "short_form_video"
            },
            {
                "content": "Mobile accounts for 54% of all ecommerce traffic. Optimize for mobile-first design with fast loading times (under 3 seconds), easy navigation, and one-click checkout options.",
                "source": "Shopify Research",
                "category": "mobile_optimization",
                "topic": "mobile_first"
            },
            {
                "content": "Personalization can increase revenue by 5-15%. Use browsing history, purchase behavior, and demographics to show relevant product recommendations and customized landing pages.",
                "source": "McKinsey & Company",
                "category": "personalization",
                "topic": "behavioral_targeting"
            },
            {
                "content": "Customer acquisition cost (CAC) should be 3x less than customer lifetime value (CLV). If CAC > CLV/3, focus on retention and upselling existing customers rather than acquiring new ones.",
                "source": "SaaS Metrics",
                "category": "customer_economics",
                "topic": "cac_clv_ratio"
            }
        ]
        
        return [
            Document(
                page_content=insight["content"],
                metadata={
                    "source": insight["source"],
                    "category": insight["category"],
                    "topic": insight["topic"],
                    "type": "marketing_insight"
                }
            )
            for insight in marketing_insights
        ]
    
    @staticmethod
    def get_sales_insights() -> List[Document]:
        """Sales strategies from top sales experts and research"""
        
        sales_insights = [
            {
                "content": "The best salespeople ask questions 70% of the time and talk 30%. Discovery is more important than pitching. Understand the customer's pain points before presenting solutions.",
                "source": "Challenger Sale",
                "category": "sales_methodology",
                "topic": "discovery_selling"
            },
            {
                "content": "Follow up is everything. 80% of sales require 5+ follow-up attempts, but 44% of salespeople give up after one follow-up. Create a systematic follow-up sequence with value-added touchpoints.",
                "source": "Sales Research",
                "category": "sales_process",
                "topic": "follow_up"
            },
            {
                "content": "Price is only an objection when value isn't clear. Focus on ROI and outcomes rather than features. Show specific dollar amounts the customer will save or earn from your solution.",
                "source": "Value Selling",
                "category": "objection_handling",
                "topic": "value_demonstration"
            },
            {
                "content": "Urgency and scarcity work when they're genuine. Limited-time offers and limited quantities can increase conversions by 30%, but avoid fake scarcity that damages trust.",
                "source": "Psychology of Persuasion",
                "category": "persuasion_techniques",
                "topic": "urgency_scarcity"
            },
            {
                "content": "The magic number for B2B sales is 6.8 touchpoints. Most prospects need multiple interactions across different channels before making a purchase decision.",
                "source": "B2B Sales Research",
                "category": "sales_cycle",
                "topic": "touchpoint_optimization"
            },
            {
                "content": "Referral customers have 16% higher lifetime value and 37% higher retention rates. Build systematic referral programs with incentives for both referrer and referee.",
                "source": "ReferralCandy Research",
                "category": "referral_marketing",
                "topic": "referral_programs"
            },
            {
                "content": "Cross-selling is 5x more cost-effective than acquiring new customers. Amazon generates 35% of revenue from cross-selling. Suggest complementary products based on purchase history.",
                "source": "Ecommerce Research",
                "category": "revenue_optimization",
                "topic": "cross_selling"
            },
            {
                "content": "Cart abandonment affects 70% of online purchases. Recover 15% of abandoned carts with email sequences that include product reminders, social proof, and limited-time discounts.",
                "source": "Baymard Institute",
                "category": "conversion_recovery",
                "topic": "cart_abandonment"
            },
            {
                "content": "Customer reviews increase purchase likelihood by 270%. Display reviews prominently and respond to all reviews (positive and negative) to build trust and show active engagement.",
                "source": "Spiegel Research",
                "category": "trust_building",
                "topic": "review_management"
            },
            {
                "content": "The best performing sales teams use CRM data to prioritize leads. Score leads based on engagement, demographics, and buying signals to focus effort on highest-probability prospects.",
                "source": "Sales Force Research",
                "category": "lead_management",
                "topic": "lead_scoring"
            }
        ]
        
        return [
            Document(
                page_content=insight["content"],
                metadata={
                    "source": insight["source"],
                    "category": insight["category"],
                    "topic": insight["topic"],
                    "type": "sales_insight"
                }
            )
            for insight in sales_insights
        ]
    
    @staticmethod
    def get_analytics_insights() -> List[Document]:
        """Data analytics and business intelligence best practices"""
        
        analytics_insights = [
            {
                "content": "Focus on leading indicators, not just lagging indicators. Track metrics like website traffic, email open rates, and social engagement to predict future sales rather than just measuring past performance.",
                "source": "Analytics Best Practices",
                "category": "metrics_strategy",
                "topic": "leading_indicators"
            },
            {
                "content": "The 80/20 rule applies to customers and products. Typically 20% of customers generate 80% of revenue, and 20% of products drive 80% of sales. Focus optimization efforts on your top performers.",
                "source": "Pareto Principle",
                "category": "business_analysis",
                "topic": "pareto_optimization"
            },
            {
                "content": "Cohort analysis reveals true customer value. Group customers by acquisition month and track their behavior over time to understand retention, lifetime value, and seasonal patterns.",
                "source": "Growth Analytics",
                "category": "customer_analytics",
                "topic": "cohort_analysis"
            },
            {
                "content": "A/B testing should have statistical significance before making decisions. Run tests for at least 2 weeks with minimum 1000 visitors per variant to get reliable results.",
                "source": "Conversion Optimization",
                "category": "testing_methodology",
                "topic": "ab_testing"
            },
            {
                "content": "Attribution modeling matters for marketing ROI. Last-click attribution undervalues top-funnel marketing. Use multi-touch attribution to understand the full customer journey.",
                "source": "Marketing Analytics",
                "category": "attribution",
                "topic": "multi_touch_attribution"
            },
            {
                "content": "Real-time dashboards should focus on actionable metrics. Include alerts for key thresholds like inventory levels, conversion rate drops, or unusual traffic patterns.",
                "source": "BI Best Practices",
                "category": "dashboard_design",
                "topic": "actionable_metrics"
            },
            {
                "content": "Seasonal analysis helps with inventory and marketing planning. Compare year-over-year performance by month, week, and day to identify patterns and optimize for peak periods.",
                "source": "Retail Analytics",
                "category": "seasonal_analysis",
                "topic": "temporal_patterns"
            },
            {
                "content": "Customer segmentation drives personalization. Use RFM analysis (Recency, Frequency, Monetary) to identify VIP customers, at-risk customers, and potential upsell opportunities.",
                "source": "Customer Analytics",
                "category": "segmentation",
                "topic": "rfm_analysis"
            },
            {
                "content": "Data quality is more important than data quantity. Clean, accurate data from fewer sources is better than massive amounts of inconsistent data. Invest in data governance.",
                "source": "Data Management",
                "category": "data_quality",
                "topic": "data_governance"
            },
            {
                "content": "Predictive analytics should complement, not replace, human judgment. Use ML models to identify patterns and opportunities, but always validate insights with business context.",
                "source": "Machine Learning Best Practices",
                "category": "predictive_analytics",
                "topic": "human_machine_collaboration"
            }
        ]
        
        return [
            Document(
                page_content=insight["content"],
                metadata={
                    "source": insight["source"],
                    "category": insight["category"],
                    "topic": insight["topic"],
                    "type": "analytics_insight"
                }
            )
            for insight in analytics_insights
        ]
    
    @staticmethod
    def get_all_expert_knowledge() -> List[Document]:
        """Get all expert knowledge combined"""
        all_documents = []
        all_documents.extend(ExpertKnowledgeBase.get_marketing_insights())
        all_documents.extend(ExpertKnowledgeBase.get_sales_insights())
        all_documents.extend(ExpertKnowledgeBase.get_analytics_insights())
        
        logger.info(f"Generated {len(all_documents)} expert knowledge documents")
        return all_documents
    
    @staticmethod
    async def get_enhanced_knowledge_with_web_content(
        include_web_content: bool = True,
        max_web_articles_per_source: int = 3
    ) -> List[Document]:
        """Get expert knowledge enhanced with fresh web content"""
        # Start with curated knowledge
        all_documents = ExpertKnowledgeBase.get_all_expert_knowledge()
        
        if include_web_content:
            try:
                from .web_content_fetcher import MarketingContentFetcher
                
                logger.info("Fetching fresh marketing content from web sources...")
                fetcher = MarketingContentFetcher()
                web_documents = await fetcher.fetch_latest_insights(
                    max_articles_per_source=max_web_articles_per_source
                )
                
                # Add web content to knowledge base
                all_documents.extend(web_documents)
                logger.info(f"Added {len(web_documents)} web articles to knowledge base")
                
            except Exception as e:
                logger.error(f"Error fetching web content: {e}")
                logger.info("Continuing with curated knowledge only")
        
        logger.info(f"Total enhanced knowledge documents: {len(all_documents)}")
        return all_documents
    
    @staticmethod
    async def refresh_knowledge_base_with_latest_insights() -> List[Document]:
        """Refresh the entire knowledge base with latest web insights"""
        try:
            from .web_content_fetcher import MarketingContentFetcher
            
            logger.info("Refreshing knowledge base with latest insights...")
            fetcher = MarketingContentFetcher()
            
            # Get fresh content
            fresh_documents = await fetcher.fetch_latest_insights(max_articles_per_source=5)
            
            # Combine with curated content
            curated_documents = ExpertKnowledgeBase.get_all_expert_knowledge()
            
            # Prioritize fresh content by putting it first
            all_documents = fresh_documents + curated_documents
            
            logger.info(f"Refreshed knowledge base with {len(fresh_documents)} fresh articles")
            logger.info(f"Total knowledge base size: {len(all_documents)} documents")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            # Fallback to curated content only
            return ExpertKnowledgeBase.get_all_expert_knowledge()