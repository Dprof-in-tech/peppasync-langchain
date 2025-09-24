import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import centralized configuration
from ..config import LLMManager, DatabaseManager

load_dotenv()
logger = logging.getLogger(__name__)

# Inventory State for LangGraph
class InventoryState(BaseModel):
    inventory_data: List[Dict] = Field(default_factory=list)
    low_stock_alerts: List[Dict] = Field(default_factory=list)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[Dict] = Field(default_factory=list)
    actions_taken: List[Dict] = Field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None

class InventoryAgent:
    """
    LangGraph-powered inventory monitoring and management agent
    Specializes in Nigerian retail inventory optimization
    """
    
    def __init__(self):
        self.llm = LLMManager.get_chat_llm()
        self.mock_data = DatabaseManager.get_mock_data()
        
        # Build the LangGraph workflow
        self.workflow = self._build_inventory_workflow()

    def _build_inventory_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for inventory monitoring"""
        
        workflow = StateGraph(InventoryState)
        
        # Add workflow nodes
        workflow.add_node("check_inventory", self._check_inventory_node)
        workflow.add_node("analyze_trends", self._analyze_trends_node)
        workflow.add_node("generate_alerts", self._generate_alerts_node)
        workflow.add_node("create_recommendations", self._create_recommendations_node)
        workflow.add_node("execute_actions", self._execute_actions_node)
        
        # Set entry point
        workflow.set_entry_point("check_inventory")
        
        # Add edges
        workflow.add_edge("check_inventory", "analyze_trends")
        workflow.add_edge("analyze_trends", "generate_alerts")
        workflow.add_edge("generate_alerts", "create_recommendations")
        workflow.add_edge("create_recommendations", "execute_actions")
        workflow.add_edge("execute_actions", END)
        
        return workflow.compile()

    async def _check_inventory_node(self, state: InventoryState) -> InventoryState:
        """Check current inventory levels"""
        try:
            logger.info("Checking inventory levels...")
            
            # Use mock data for demonstration
            inventory_data = self.mock_data['inventory_data']
            
            # Add some dynamic elements
            for item in inventory_data:
                # Simulate some randomness in current stock
                if item['product_name'] == 'iPhone 15 Pro':
                    item['current_stock'] = max(0, item['current_stock'] + (hash(str(time.time())) % 10 - 5))
            
            state.inventory_data = inventory_data
            
            logger.info(f"Checked {len(inventory_data)} products, found {len([i for i in inventory_data if i['current_stock'] < i['reorder_level']])} low stock items")
            
        except Exception as e:
            logger.error(f"Error checking inventory: {e}")
            state.error = f"Inventory check failed: {str(e)}"
        
        return state

    async def _analyze_trends_node(self, state: InventoryState) -> InventoryState:
        """Analyze inventory trends and velocity"""
        try:
            logger.info("Analyzing inventory trends...")
            
            # Calculate basic trend metrics
            total_items = len(state.inventory_data)
            low_stock_count = len([i for i in state.inventory_data if i['current_stock'] < i['reorder_level']])
            critical_stock_count = len([i for i in state.inventory_data if i['current_stock'] < (i['reorder_level'] * 0.5)])
            
            # Determine trend velocity
            if critical_stock_count > total_items * 0.3:
                velocity = "critical"
            elif low_stock_count > total_items * 0.4:
                velocity = "high"
            elif low_stock_count > total_items * 0.2:
                velocity = "moderate"
            else:
                velocity = "stable"
            
            state.trend_analysis = {
                'velocity': velocity,
                'total_items': total_items,
                'low_stock_count': low_stock_count,
                'critical_stock_count': critical_stock_count,
                'trend_direction': 'declining' if low_stock_count > 0 else 'stable',
                'analysis_timestamp': int(time.time())
            }
            
            logger.info(f"Trend analysis complete: velocity={velocity}, {low_stock_count} items need attention")
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            state.error = f"Trend analysis failed: {str(e)}"
        
        return state

    async def _generate_alerts_node(self, state: InventoryState) -> InventoryState:
        """Generate inventory alerts based on stock levels"""
        try:
            logger.info("Generating inventory alerts...")
            
            alerts = []
            
            for item in state.inventory_data:
                current_stock = item['current_stock']
                reorder_level = item['reorder_level']
                
                if current_stock <= 0:
                    priority = "CRITICAL"
                    alert_type = "STOCK_OUT"
                elif current_stock < reorder_level * 0.5:
                    priority = "CRITICAL"
                    alert_type = "EXTREMELY_LOW_STOCK"
                elif current_stock < reorder_level:
                    priority = "HIGH"
                    alert_type = "LOW_STOCK"
                else:
                    continue
                
                alert = {
                    'alert_id': f"INV_{item['product_id']}_{int(time.time())}",
                    'product_id': item['product_id'],
                    'product_name': item['product_name'],
                    'alert_type': alert_type,
                    'priority': priority,
                    'current_stock': current_stock,
                    'reorder_level': reorder_level,
                    'recommended_quantity': max(reorder_level * 2, 10),
                    'estimated_cost': item['unit_cost'] * max(reorder_level * 2, 10),
                    'supplier': item.get('supplier', 'Unknown'),
                    'timestamp': int(time.time())
                }
                
                alerts.append(alert)
            
            state.low_stock_alerts = alerts
            
            logger.info(f"Generated {len(alerts)} inventory alerts")
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            state.error = f"Alert generation failed: {str(e)}"
        
        return state

    async def _create_recommendations_node(self, state: InventoryState) -> InventoryState:
        """Create AI-powered inventory recommendations"""
        try:
            logger.info("Creating inventory recommendations...")
            
            # Prepare context for AI recommendations
            inventory_context = {
                'total_items': len(state.inventory_data),
                'alerts_count': len(state.low_stock_alerts),
                'trend_velocity': state.trend_analysis.get('velocity', 'unknown'),
                'critical_items': [alert for alert in state.low_stock_alerts if alert['priority'] == 'CRITICAL']
            }
            
            system_message = """You are an expert inventory management consultant for Nigerian retail businesses. 
            Provide practical, actionable inventory recommendations based on current stock levels and trends.
            Consider local supply chain dynamics, cash flow implications, and seasonal factors.
            
            Format responses as JSON with: recommendation_type, description, priority, estimated_impact, timeline"""
            
            human_message = f"""
            Analyze this inventory situation and provide recommendations:
            
            Context: {json.dumps(inventory_context, indent=2)}
            
            Critical alerts: {len([a for a in state.low_stock_alerts if a['priority'] == 'CRITICAL'])}
            
            Generate 3-5 specific recommendations for inventory optimization.
            """
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse AI recommendations (simplified for demo)
            recommendations = [
                {
                    'recommendation_id': f"REC_{int(time.time())}_{i}",
                    'type': 'REORDER_OPTIMIZATION',
                    'description': f"Optimize reorder quantities for {len(state.low_stock_alerts)} low-stock items",
                    'priority': 'HIGH',
                    'estimated_impact': 'Prevent stockouts, maintain service levels',
                    'timeline': '1-2 weeks',
                    'ai_reasoning': response.content[:200] + "..." if len(response.content) > 200 else response.content
                }
                for i in range(min(5, len(state.low_stock_alerts) + 1))
            ]
            
            state.recommendations = recommendations
            
            logger.info(f"Created {len(recommendations)} inventory recommendations")
            
        except Exception as e:
            logger.error(f"Error creating recommendations: {e}")
            state.error = f"Recommendation creation failed: {str(e)}"
            # Fallback recommendations
            state.recommendations = [{
                'recommendation_id': f"REC_FALLBACK_{int(time.time())}",
                'type': 'BASIC_REORDER',
                'description': 'Review and reorder low stock items',
                'priority': 'MEDIUM'
            }]
        
        return state

    async def _execute_actions_node(self, state: InventoryState) -> InventoryState:
        """Execute automated inventory actions"""
        try:
            logger.info("Executing inventory actions...")
            
            actions_taken = []
            
            # Action 1: Log critical inventory events
            critical_items = [alert for alert in state.low_stock_alerts if alert['priority'] == 'CRITICAL']
            for item in critical_items:
                action = {
                    'action_id': f"ACTION_{int(time.time())}_{item['product_id']}",
                    'action_type': 'CRITICAL_INVENTORY_LOG',
                    'details': item,
                    'timestamp': int(time.time())
                }
                actions_taken.append(action)
                
                # Log as warning for monitoring systems
                logger.warning(f"CRITICAL_INVENTORY_EVENT: {json.dumps({
                    'event_type': 'CRITICAL_INVENTORY',
                    'product_id': item['product_id'],
                    'product_name': item['product_name'],
                    'current_stock': item['current_stock'],
                    'estimated_cost': item['estimated_cost'],
                    'supplier': item['supplier'],
                    'timestamp': int(time.time())
                })}")
            
            # Action 2: Generate summary alerts for management
            if len(state.low_stock_alerts) > 0:
                summary_action = {
                    'action_id': f"ACTION_SUMMARY_{int(time.time())}",
                    'action_type': 'INVENTORY_SUMMARY_ALERT',
                    'details': {
                        'total_alerts': len(state.low_stock_alerts),
                        'critical_count': len(critical_items),
                        'estimated_reorder_cost': sum([alert['estimated_cost'] for alert in state.low_stock_alerts])
                    },
                    'timestamp': int(time.time())
                }
                actions_taken.append(summary_action)
                
                logger.info(f"INVENTORY_SUMMARY_ALERT: {json.dumps({
                    'action_type': 'INVENTORY_SUMMARY_ALERT',
                    'critical_items_count': len(critical_items),
                    'total_estimated_cost': summary_action['details']['estimated_reorder_cost'],
                    'timestamp': int(time.time())
                })}")
            
            # Action 3: Create procurement queue
            if state.recommendations:
                procurement_items = []
                for alert in state.low_stock_alerts:
                    procurement_items.append({
                        'product_id': alert['product_id'],
                        'product_name': alert['product_name'],
                        'recommended_quantity': alert['recommended_quantity'],
                        'estimated_cost': alert['estimated_cost'],
                        'priority': alert['priority']
                    })
                
                if procurement_items:
                    procurement_action = {
                        'action_id': f"ACTION_PROCUREMENT_{int(time.time())}",
                        'action_type': 'PROCUREMENT_QUEUE_UPDATE',
                        'details': {
                            'items': procurement_items[:3],  # Top 3 priority items
                            'total_items': len(procurement_items),
                            'total_estimated_cost': sum([item['estimated_cost'] for item in procurement_items])
                        },
                        'timestamp': int(time.time())
                    }
                    actions_taken.append(procurement_action)
                    
                    logger.info(f"PROCUREMENT_QUEUE: {json.dumps({
                        'action_type': 'PROCUREMENT_QUEUE_UPDATE',
                        'items': procurement_items[:3],
                        'total_items': len(procurement_items),
                        'total_estimated_cost': procurement_action['details']['total_estimated_cost'],
                        'timestamp': int(time.time())
                    })}")
            
            state.actions_taken = actions_taken
            state.status = "completed"
            
            logger.info(f"Executed {len(actions_taken)} inventory actions")
            
        except Exception as e:
            logger.error(f"Error executing actions: {e}")
            state.error = f"Action execution failed: {str(e)}"
            state.status = "error"
        
        return state

    async def run_inventory_monitoring(self) -> Dict[str, Any]:
        """Main method to run inventory monitoring workflow"""
        try:
            # Initialize state
            state = InventoryState()
            
            # Run the LangGraph workflow
            result = await self.workflow.ainvoke(state)
            
            # Handle different return types from LangGraph versions
            if isinstance(result, dict):
                # Newer LangGraph versions return dict
                final_state = result
            else:
                # Older versions might return state object
                final_state = result.__dict__ if hasattr(result, '__dict__') else result
            
            # Format response
            return {
                "status": final_state.get('status', 'completed'),
                "timestamp": int(time.time()),
                "inventory_status": {
                    "data": final_state.get('inventory_data', []),
                    "low_stock_alerts": final_state.get('low_stock_alerts', [])
                },
                "trend_analysis": final_state.get('trend_analysis', {}),
                "recommendations": final_state.get('recommendations', []),
                "actions_taken": final_state.get('actions_taken', []),
                "summary": {
                    "products_checked": len(final_state.get('inventory_data', [])),
                    "critical_alerts": len([a for a in final_state.get('low_stock_alerts', []) if a.get('priority') == 'CRITICAL']),
                    "actions_executed": len(final_state.get('actions_taken', []))
                },
                "error": final_state.get('error', None)
            }
            
        except Exception as e:
            logger.error(f"Error in inventory monitoring workflow: {e}")
            return {
                "status": "error",
                "timestamp": int(time.time()),
                "error": str(e),
                "message": "Inventory monitoring failed"
            }

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current inventory agent status"""
        return {
            "agent": "InventoryAgent",
            "status": "active",
            "capabilities": [
                "Real-time inventory monitoring",
                "Low stock alerting",
                "Reorder recommendations",
                "Supplier coordination",
                "Cash flow impact analysis"
            ],
            "last_updated": int(time.time())
        }