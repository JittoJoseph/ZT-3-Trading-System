"""
Notifications Module for ZT-3 Trading System.

This module handles notifications to Discord channels for
trade alerts, system status, and error reporting.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class NotificationColor(Enum):
    """Color codes for Discord message embeds."""
    SUCCESS = 0x00FF00  # Green
    ERROR = 0xFF0000    # Red
    WARNING = 0xFFAA00  # Orange
    INFO = 0x00AAFF     # Blue
    TRADE_BUY = 0x00FF68  # Light Green
    TRADE_SELL = 0xFF0A5A  # Pink
    PERFORMANCE = 0xAA00FF  # Purple
    SIGNAL = 0xFFFF00    # Yellow

class NotificationChannel(Enum):
    """Types of notification channels."""
    TRADE_ALERTS = "trade_alerts"
    PERFORMANCE = "performance"
    SIGNALS = "signals"
    SYSTEM_STATUS = "system_status"

class DiscordNotifier:
    """
    Discord webhook notification system for the ZT-3 Trading System.
    
    Sends notifications to configured Discord channels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Discord notifier with webhook configuration.
        
        Args:
            config: Configuration containing Discord webhook URLs and settings
        """
        self.config = config.get('notifications', {})
        self.webhooks = self.config.get('discord_webhooks', {})
        self.enabled_levels = self.config.get('notification_levels', {
            'trade_alerts': True,
            'performance': True,
            'signals': True,
            'system_status': True
        })
        
        self.username = self.config.get('discord_username', 'ZT-3 Trading Bot')
        self.rate_limit_delay = 1.0  # Seconds to wait between API calls to avoid rate limiting
        self.last_notification_time = {}  # Channel -> timestamp of last notification
        
        logger.info("Discord notifier initialized")
    
    def _get_webhook_url(self, channel: NotificationChannel) -> Optional[str]:
        """
        Get the webhook URL for a specific notification channel.
        
        Args:
            channel: Notification channel enum
            
        Returns:
            Webhook URL or None if not configured
        """
        if channel.value in self.webhooks:
            return self.webhooks[channel.value]
        return None
    
    def _is_enabled(self, channel: NotificationChannel) -> bool:
        """
        Check if a notification channel is enabled.
        
        Args:
            channel: Notification channel enum
            
        Returns:
            True if notifications for this channel are enabled
        """
        return self.enabled_levels.get(channel.value, False)
    
    def _respect_rate_limit(self, channel: NotificationChannel) -> None:
        """
        Respect Discord rate limits by waiting if needed.
        
        Args:
            channel: Notification channel enum
        """
        channel_key = channel.value
        current_time = time.time()
        
        if channel_key in self.last_notification_time:
            elapsed = current_time - self.last_notification_time[channel_key]
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_notification_time[channel_key] = time.time()
    
    def send_notification(self, 
                         channel: NotificationChannel, 
                         content: str,
                         title: Optional[str] = None,
                         color: NotificationColor = NotificationColor.INFO,
                         fields: Optional[List[Dict[str, str]]] = None,
                         footer: Optional[str] = None) -> bool:
        """
        Send a notification to a Discord channel.
        
        Args:
            channel: Notification channel to send to
            content: Message content
            title: Optional title for the embed
            color: Color for the embed
            fields: Optional list of fields for the embed
            footer: Optional footer text
            
        Returns:
            True if notification was sent successfully
        """
        # Check if this notification channel is enabled
        if not self._is_enabled(channel):
            logger.debug(f"Notifications for channel {channel.value} are disabled")
            return False
        
        # Get webhook URL
        webhook_url = self._get_webhook_url(channel)
        if not webhook_url:
            logger.warning(f"No webhook URL configured for channel {channel.value}")
            return False
        
        # Respect rate limits
        self._respect_rate_limit(channel)
        
        # Build Discord message
        message = {
            "username": self.username,
            "embeds": [{
                "description": content,
                "color": color.value
            }]
        }
        
        # Add title if provided
        if title:
            message["embeds"][0]["title"] = title
        
        # Add fields if provided
        if fields:
            message["embeds"][0]["fields"] = fields
        
        # Add footer with timestamp if provided
        if footer:
            message["embeds"][0]["footer"] = {
                "text": f"{footer} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        else:
            message["embeds"][0]["footer"] = {
                "text": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        
        # Try to send notification
        try:
            response = requests.post(
                webhook_url,
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 204:
                logger.debug(f"Notification sent to {channel.value}")
                return True
            else:
                logger.warning(f"Failed to send notification to {channel.value}: {response.status_code} - {response.text}")
                return False
                
        except RequestException as e:
            logger.error(f"Error sending notification to {channel.value}: {e}")
            return False
    
    def send_trade_alert(self, 
                        trade_type: str, 
                        symbol: str, 
                        price: float,
                        quantity: int,
                        take_profit: Optional[float] = None,
                        stop_loss: Optional[float] = None,
                        pnl: Optional[float] = None,
                        pnl_percent: Optional[float] = None,
                        reason: Optional[str] = None) -> bool:
        """
        Send a trade alert notification.
        
        Args:
            trade_type: Type of trade (BUY/SELL)
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            take_profit: Optional take profit level
            stop_loss: Optional stop loss level
            pnl: Optional realized P&L for exits
            pnl_percent: Optional P&L percentage for exits
            reason: Optional exit reason
            
        Returns:
            True if notification was sent successfully
        """
        is_buy = trade_type.upper() == "BUY"
        color = NotificationColor.TRADE_BUY if is_buy else NotificationColor.TRADE_SELL
        
        # Format content based on trade type
        if is_buy:
            title = f"🟢 BUY SIGNAL - {symbol}"
            content = f"Entry: ₹{price:.2f}\nQuantity: {quantity} shares"
            
            # Add take profit/stop loss if available
            if take_profit:
                content += f"\nTake Profit: ₹{take_profit:.2f}"
            if stop_loss:
                content += f"\nStop Loss: ₹{stop_loss:.2f}"
                
            # Calculate capital used
            capital_used = price * quantity
            content += f"\nCapital Used: ₹{capital_used:.2f}"
            
        else:
            title = f"🔴 SELL SIGNAL - {symbol}"
            content = f"Exit: ₹{price:.2f}\nQuantity: {quantity} shares"
            
            # Add P&L info if available
            if pnl is not None:
                content += f"\nP&L: {'+'if pnl >= 0 else ''}₹{pnl:.2f}"
            if pnl_percent is not None:
                content += f" ({'+' if pnl_percent >= 0 else ''}{pnl_percent:.2f}%)"
                
            # Add exit reason if available
            if reason:
                content += f"\nReason: {reason}"
        
        # Add timestamp
        content += f"\nTime: {datetime.now().strftime('%H:%M:%S')}"
        
        return self.send_notification(
            channel=NotificationChannel.TRADE_ALERTS,
            title=title,
            content=content,
            color=color
        )
    
    def send_signal_alert(self,
                         symbol: str,
                         conditions: Dict[str, Any],
                         price: float) -> bool:
        """
        Send a signal detection notification.
        
        Args:
            symbol: Trading symbol
            conditions: Dictionary of signal conditions
            price: Current price
            
        Returns:
            True if notification was sent successfully
        """
        # Format fields for conditions
        fields = []
        for name, value in conditions.items():
            if isinstance(value, bool):
                value_str = "✅" if value else "❌"
            elif isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            fields.append({
                "name": name,
                "value": value_str,
                "inline": True
            })
        
        return self.send_notification(
            channel=NotificationChannel.SIGNALS,
            title=f"📝 SIGNAL DETECTED - {symbol}",
            content=f"Current Price: ₹{price:.2f}",
            color=NotificationColor.SIGNAL,
            fields=fields
        )
    
    def send_status_update(self, 
                          status: str, 
                          details: Optional[str] = None,
                          color: NotificationColor = NotificationColor.INFO) -> bool:
        """
        Send a system status notification.
        
        Args:
            status: Status message
            details: Optional additional details
            color: Notification color
            
        Returns:
            True if notification was sent successfully
        """
        content = status
        if details:
            content += f"\n{details}"
        
        return self.send_notification(
            channel=NotificationChannel.SYSTEM_STATUS,
            content=content,
            color=color,
            footer="System Status"
        )
    
    def send_error_alert(self, 
                        error_message: str, 
                        details: Optional[str] = None) -> bool:
        """
        Send an error notification.
        
        Args:
            error_message: Error message
            details: Optional error details
            
        Returns:
            True if notification was sent successfully
        """
        content = f"⚠️ ERROR: {error_message}"
        if details:
            content += f"\n```\n{details}\n```"
        
        return self.send_notification(
            channel=NotificationChannel.SYSTEM_STATUS,
            content=content,
            color=NotificationColor.ERROR,
            footer="Error Report"
        )
    
    def send_performance_report(self, performance: Dict[str, Any]) -> bool:
        """
        Send a performance report notification.
        
        Args:
            performance: Dictionary with performance metrics
            
        Returns:
            True if notification was sent successfully
        """
        # Format performance data
        fields = []
        
        # Add trade counts
        win_count = performance.get("win_trades", 0)
        loss_count = performance.get("loss_trades", 0)
        total_trades = performance.get("total_trades", 0)
        win_rate = performance.get("win_rate", 0.0)
        
        trades_field = f"{total_trades} ({win_count} wins, {loss_count} losses)"
        fields.append({
            "name": "Trades",
            "value": trades_field,
            "inline": True
        })
        
        fields.append({
            "name": "Win Rate",
            "value": f"{win_rate:.1f}%",
            "inline": True
        })
        
        # Add P&L
        net_pnl = performance.get("net_pnl", 0.0)
        net_pnl_percent = performance.get("net_pnl_percent", 0.0)
        
        fields.append({
            "name": "Net P&L",
            "value": f"{'+' if net_pnl >= 0 else ''}₹{net_pnl:.2f} ({'+' if net_pnl_percent >= 0 else ''}{net_pnl_percent:.2f}%)",
            "inline": True
        })
        
        # Add drawdown
        drawdown = performance.get("max_drawdown", 0.0)
        drawdown_percent = performance.get("max_drawdown_percent", 0.0)
        
        fields.append({
            "name": "Max Drawdown",
            "value": f"₹{drawdown:.2f} ({drawdown_percent:.2f}%)",
            "inline": True
        })
        
        # Add best/worst trades
        best_trade = performance.get("best_trade", {})
        worst_trade = performance.get("worst_trade", {})
        
        if best_trade:
            fields.append({
                "name": "Best Trade",
                "value": f"{best_trade.get('symbol', '')} +₹{best_trade.get('pnl', 0.0):.2f}",
                "inline": True
            })
        
        if worst_trade:
            fields.append({
                "name": "Worst Trade",
                "value": f"{worst_trade.get('symbol', '')} ₹{worst_trade.get('pnl', 0.0):.2f}",
                "inline": True
            })
        
        # Add title based on report type (daily vs. overall)
        is_daily = "date" in performance
        title = f"📊 DAILY SUMMARY - {performance['date']}" if is_daily else "📊 OVERALL PERFORMANCE"
        
        return self.send_notification(
            channel=NotificationChannel.PERFORMANCE,
            title=title,
            content="",  # Empty content since we're using fields
            color=NotificationColor.PERFORMANCE,
            fields=fields
        )