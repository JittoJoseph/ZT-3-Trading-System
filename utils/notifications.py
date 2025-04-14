"""
Notification utilities for ZT-3 Trading System.

This module handles sending notifications to Discord webhooks.
"""

import json
import logging
import requests
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications that can be sent."""
    TRADE_ALERT = "trade_alerts"
    PERFORMANCE = "performance"
    SIGNAL = "signals"
    SYSTEM_STATUS = "system_status"

class NotificationManager:
    """
    Manages sending notifications to configured channels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification manager with configuration.
        
        Args:
            config: Configuration dictionary containing webhook URLs and notification levels.
        """
        self.webhooks = config.get('discord_webhooks', {})
        self.levels = config.get('notification_levels', {})
        
    def send_discord_notification(self, 
                                 notification_type: NotificationType, 
                                 title: str, 
                                 message: str,
                                 color: int = 0x00FF00,
                                 fields: Optional[Dict[str, str]] = None) -> bool:
        """
        Send a notification to Discord webhook.
        
        Args:
            notification_type: Type of notification (determines which webhook to use)
            title: Title of the message
            message: Content of the message
            color: Color of the Discord embed (hexadecimal)
            fields: Optional dictionary of field name -> value to add to the embed
            
        Returns:
            bool: True if the notification was sent successfully, False otherwise.
        """
        # Check if this notification type is enabled
        type_name = notification_type.value
        if not self.levels.get(type_name, False):
            logger.debug(f"Notification type {type_name} is disabled.")
            return False
            
        # Get webhook URL for this notification type
        webhook_url = self.webhooks.get(type_name)
        if not webhook_url:
            logger.warning(f"No webhook URL configured for {type_name}")
            return False
            
        # Prepare the payload
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        # Add fields if provided
        if fields:
            embed_fields = []
            for name, value in fields.items():
                embed_fields.append({
                    "name": name,
                    "value": value,
                    "inline": True
                })
            embed["fields"] = embed_fields
            
        payload = {
            "embeds": [embed]
        }
        
        # Send the notification
        try:
            response = requests.post(
                webhook_url, 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.debug(f"Sent {notification_type.name} notification: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    def send_trade_alert(self, action: str, symbol: str, price: float, quantity: int, 
                        reason: str = "", additional_info: Optional[Dict[str, str]] = None) -> bool:
        """
        Send a trade alert notification.
        
        Args:
            action: The trade action (BUY, SELL)
            symbol: Trading symbol
            price: Execution price
            quantity: Number of shares
            reason: Reason for the trade
            additional_info: Any additional information to include
            
        Returns:
            bool: True if notification was sent successfully
        """
        title = f"{action} {symbol} ALERT"
        message = f"Executed {action} for {quantity} shares of {symbol} @ ₹{price:.2f}"
        if reason:
            message += f"\nReason: {reason}"
            
        # Set color based on action
        color = 0x00FF00 if action == "BUY" else 0xFF0000
        
        # Prepare fields
        fields = {
            "Symbol": symbol,
            "Action": action,
            "Quantity": str(quantity),
            "Price": f"₹{price:.2f}",
        }
        
        if additional_info:
            fields.update(additional_info)
            
        return self.send_discord_notification(
            NotificationType.TRADE_ALERT, 
            title, 
            message, 
            color, 
            fields
        )
    
    def send_signal_alert(self, symbol: str, signal_type: str, 
                         indicators: Dict[str, Any]) -> bool:
        """
        Send a signal alert notification.
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal (ENTRY, EXIT, etc.)
            indicators: Dictionary of indicator values
            
        Returns:
            bool: True if notification was sent successfully
        """
        title = f"{signal_type} SIGNAL for {symbol}"
        message = f"Strategy detected a {signal_type} signal for {symbol}"
        
        # Format indicators for display
        fields = {}
        for name, value in indicators.items():
            if isinstance(value, float):
                fields[name] = f"{value:.4f}"
            else:
                fields[name] = str(value)
                
        # Set color based on signal type
        color = 0x00FF00 if signal_type == "ENTRY" else 0xFF0000
        
        return self.send_discord_notification(
            NotificationType.SIGNAL, 
            title, 
            message, 
            color, 
            fields
        )
    
    def send_system_status(self, status: str, message: str, 
                          is_error: bool = False) -> bool:
        """
        Send a system status notification.
        
        Args:
            status: Status message (e.g., "STARTED", "ERROR", etc.)
            message: Detailed status message
            is_error: Whether this status represents an error
            
        Returns:
            bool: True if notification was sent successfully
        """
        title = f"SYSTEM {status}"
        color = 0xFF0000 if is_error else 0x0000FF
        
        return self.send_discord_notification(
            NotificationType.SYSTEM_STATUS, 
            title, 
            message, 
            color
        )
    
    def send_performance_update(self, daily_pnl: float, total_pnl: float,
                               trade_count: int, win_rate: float,
                               additional_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a performance update notification.
        
        Args:
            daily_pnl: Profit/loss for the day
            total_pnl: Total profit/loss
            trade_count: Number of trades executed
            win_rate: Winning trades percentage
            additional_metrics: Any additional metrics to include
            
        Returns:
            bool: True if notification was sent successfully
        """
        title = "PERFORMANCE UPDATE"
        message = f"Daily P&L: ₹{daily_pnl:.2f} | Total P&L: ₹{total_pnl:.2f}"
        
        fields = {
            "Daily P&L": f"₹{daily_pnl:.2f}",
            "Total P&L": f"₹{total_pnl:.2f}",
            "Trades": str(trade_count),
            "Win Rate": f"{win_rate:.1f}%"
        }
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, float):
                    fields[key] = f"{value:.2f}"
                else:
                    fields[key] = str(value)
        
        # Set color based on daily P&L
        color = 0x00FF00 if daily_pnl >= 0 else 0xFF0000
        
        return self.send_discord_notification(
            NotificationType.PERFORMANCE, 
            title, 
            message, 
            color, 
            fields
        )