from langchain.tools import BaseTool
from typing import Dict, List, Any
import logging
import time

logger = logging.getLogger(__name__)

class AlertTool(BaseTool):
    """Generate alerts for critical business conditions"""
    name: str = "alert_generation"
    description: str = "Generate alerts and notifications for critical business conditions"

    def _run(self, data: List[Dict], alert_type: str, thresholds: Dict[str, Any] = None) -> List[Dict]:
        """Generate alerts based on data and thresholds"""
        try:
            alerts = []
            thresholds = thresholds or {}
            
            if alert_type == "inventory_alerts":
                for item in data:
                    current_stock = item.get('current_stock', 0)
                    reorder_level = item.get('reorder_level', 0)
                    
                    if current_stock <= 0:
                        priority = "CRITICAL"
                        message = f"STOCKOUT: {item.get('product_name')} is out of stock"
                    elif current_stock <= reorder_level * 0.5:
                        priority = "HIGH"
                        message = f"LOW STOCK: {item.get('product_name')} has only {current_stock} units left"
                    elif current_stock <= reorder_level:
                        priority = "MEDIUM"
                        message = f"REORDER NEEDED: {item.get('product_name')} below reorder level"
                    else:
                        continue
                    
                    alerts.append({
                        "alert_id": f"INV_{item.get('product_name', 'unknown')}_{int(time.time())}",
                        "type": "INVENTORY",
                        "priority": priority,
                        "message": message,
                        "data": item,
                        "timestamp": int(time.time())
                    })
                    
            elif alert_type == "marketing_alerts":
                roas_threshold = thresholds.get('min_roas', 2.0)
                for campaign in data:
                    roas = campaign.get('roas', 0)
                    if roas < roas_threshold:
                        alerts.append({
                            "alert_id": f"CAMP_{campaign.get('campaign_id')}_{int(time.time())}",
                            "type": "MARKETING",
                            "priority": "HIGH" if roas < 1.5 else "MEDIUM",
                            "message": f"Low ROAS: {campaign.get('campaign_name')} has ROAS of {roas}",
                            "data": campaign,
                            "timestamp": int(time.time())
                        })
            
            elif alert_type == "sales_alerts":
                # Revenue decline alerts
                if len(data) >= 2:
                    latest = data[-1]
                    previous = data[-2]
                    if latest.get('revenue', 0) < previous.get('revenue', 0) * 0.9:  # 10% decline
                        alerts.append({
                            "alert_id": f"SALES_DECLINE_{int(time.time())}",
                            "type": "SALES",
                            "priority": "HIGH",
                            "message": f"Revenue declined from ₦{previous.get('revenue', 0):,} to ₦{latest.get('revenue', 0):,}",
                            "data": {"current": latest, "previous": previous},
                            "timestamp": int(time.time())
                        })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return [{"error": str(e)}]

    async def _arun(self, data: List[Dict], alert_type: str, thresholds: Dict[str, Any] = None) -> List[Dict]:
        """Async version of alert generation"""
        return self._run(data, alert_type, thresholds)