"""
Notification utilities for ZT-3 Trading System.

This module provides notification functionality including Discord webhooks
for various types of notifications.
"""

import logging
import os
import json
import requests
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from strategy import Signal, SignalType

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Handles notifications for the ZT-3 Trading System.
    
    This includes Discord webhook notifications for various events
    like trade alerts, system status, errors, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification manager with config.
        
        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config
        notif_config = config.get('notifications', {})
        
        # Check if Discord notifications are enabled
        self.enabled = notif_config.get('discord', {}).get('enabled', False)
        
        # Get webhook URLs from config or environment variables
        self.webhooks = {}
        discord_config = notif_config.get('discord', {}).get('webhooks', {})
        
        # Try to get webhooks from config first, then from environment variables
        webhook_types = ['trade_alerts', 'performance', 'signals', 'system_status', 'backtest_results']
        for webhook_type in webhook_types:
            webhook_url = discord_config.get(webhook_type) or os.environ.get(f"DISCORD_WEBHOOK_{webhook_type.upper()}")
            if webhook_url:
                self.webhooks[webhook_type] = webhook_url
        
        # Get notification levels
        self.notification_levels = notif_config.get('notification_levels', {})
        
        logger.debug(f"NotificationManager initialized with {len(self.webhooks)} webhooks")
        
    def send_system_notification(self, title: str, message: str, level: str = "info") -> bool:
        """
        Send a system notification.
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level (info, warning, error, success)
            
        Returns:
            True if notification was sent, False otherwise
        """
        if not self.enabled or not self.notification_levels.get('system_status', True):
            logger.debug(f"System notifications disabled, would send: {title} - {message}")
            return False
            
        webhook_url = self.webhooks.get('system_status')
        if not webhook_url:
            logger.warning("No system_status webhook URL configured")
            return False
        
        # Set up the message
        color = {
            "info": 0x3498db,     # Blue
            "warning": 0xe67e22,  # Orange
            "error": 0xe74c3c,    # Red
            "success": 0x2ecc71   # Green
        }.get(level, 0x95a5a6)    # Default: Grey
        
        # Create emoji prefix based on level
        emoji_prefix = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }.get(level, "🔔")
        
        # Create rich embed for Discord
        embed = {
            "title": f"{emoji_prefix} {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        payload = {
            "embeds": [embed]
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
            return False
    
    def send_signal_notification(self, signal: Signal) -> bool:
        """
        Send a signal notification.
        
        Args:
            signal: Signal object from strategy
            
        Returns:
            True if notification was sent, False otherwise
        """
        if not self.enabled or not self.notification_levels.get('signals', True):
            return False
            
        webhook_url = self.webhooks.get('signals')
        if not webhook_url:
            logger.warning("No signals webhook URL configured")
            return False
        
        # Determine color based on signal type
        color = 0x2ecc71 if signal.signal_type == SignalType.ENTRY else 0xe74c3c  # Green for entry, red for exit
        
        # Format the message
        title = f"{'🟢 Entry' if signal.signal_type == SignalType.ENTRY else '🔴 Exit'} Signal - {signal.symbol}"
        
        description = f"**Price:** ₹{signal.price:.2f}\n"
        description += f"**Time:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if signal.signal_type == SignalType.ENTRY:
            if signal.take_profit_level:
                description += f"**Take Profit:** ₹{signal.take_profit_level:.2f}\n"
            if signal.stop_loss_level:
                description += f"**Stop Loss:** ₹{signal.stop_loss_level:.2f}\n"
        elif signal.signal_type == SignalType.EXIT and signal.exit_reason:
            description += f"**Exit Reason:** {signal.exit_reason.value}\n"
        
        # Create embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        payload = {
            "embeds": [embed]
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send signal notification: {e}")
            return False
    
    def send_trade_execution_notification(self, trade_result: Dict[str, Any]) -> bool:
        """
        Send notification about executed trade.
        
        Args:
            trade_result: Dictionary with trade details
            
        Returns:
            True if notification was sent, False otherwise
        """
        if not self.enabled or not self.notification_levels.get('trade_alerts', True):
            return False
            
        webhook_url = self.webhooks.get('trade_alerts')
        if not webhook_url:
            logger.warning("No trade_alerts webhook URL configured")
            return False
        
        # Determine if this is a buy or sell
        is_buy = trade_result.get('type', '').upper() == 'BUY'
        color = 0x2ecc71 if is_buy else 0xe74c3c  # Green for buy, red for sell
        
        # Format the message
        title = f"{'🟢 BUY' if is_buy else '🔴 SELL'} - {trade_result.get('symbol', 'Unknown')}"
        
        description = f"**Price:** ₹{trade_result.get('price', 0):.2f}\n"
        description += f"**Quantity:** {trade_result.get('quantity', 0)}\n"
        description += f"**Value:** ₹{trade_result.get('value', 0):.2f}\n"
        description += f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if is_buy:
            if trade_result.get('take_profit'):
                description += f"**Take Profit:** ₹{trade_result.get('take_profit'):.2f}\n"
            if trade_result.get('stop_loss'):
                description += f"**Stop Loss:** ₹{trade_result.get('stop_loss'):.2f}\n"
        else:
            if trade_result.get('pnl'):
                pnl = trade_result.get('pnl', 0)
                pnl_percent = trade_result.get('pnl_percent', 0)
                pnl_emoji = "🟢" if pnl > 0 else "🔴"
                description += f"**P&L:** {pnl_emoji} ₹{pnl:.2f} ({pnl_percent:.2f}%)\n"
            
            if trade_result.get('exit_reason'):
                description += f"**Exit Reason:** {trade_result.get('exit_reason')}\n"
        
        # Create embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        payload = {
            "embeds": [embed]
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send trade execution notification: {e}")
            return False
    
    def send_backtest_results(self, metrics: Dict[str, Any]) -> bool:
        """
        Send backtest results summary to Discord.

        Args:
            metrics: Dictionary containing backtest performance metrics.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self.webhooks.get('backtest_results'):
            logger.warning("Discord webhook URL not configured. Cannot send backtest results.")
            return False

        try:
            # --- Create a more structured embed ---
            strategy_name = metrics.get('strategy_name', 'Unknown Strategy')
            symbols = metrics.get('symbol', 'N/A')
            start_date = metrics.get('start_date', 'N/A')
            end_date = metrics.get('end_date', 'N/A')
            duration = metrics.get('duration_days', 'N/A')

            # --- Build fields list with "Name: Value" format ---
            fields_list = []

            # Helper function to add a field if the metric exists
            def add_metric_field(name, key, format_str="{:.2f}", unit=""):
                value = metrics.get(key)
                if value is not None:
                    fields_list.append({
                        "name": "\u200B", # Zero-width space for name
                        "value": f"**{name}:** {format_str.format(value)}{unit}"
                    })

            # Add metrics in desired order
            add_metric_field("Total Return", 'total_return_percent', unit="%")
            add_metric_field("Max Drawdown", 'max_drawdown_percent', unit="%")
            add_metric_field("Sharpe Ratio", 'sharpe_ratio')
            
            # Add a separator line (optional) - using markdown
            # fields_list.append({"name": "\u200B", "value": "---"}) 
            
            add_metric_field("Total Trades", 'total_trades', format_str="{:d}")
            add_metric_field("Win Rate", 'win_rate_percent', unit="%")
            add_metric_field("Profit Factor", 'profit_factor')
            
            # Custom format for Avg Profit/Loss
            avg_profit = metrics.get('avg_profit')
            avg_loss = metrics.get('avg_loss')
            if avg_profit is not None and avg_loss is not None:
                 fields_list.append({
                     "name": "\u200B",
                     "value": f"**Avg Profit/Loss:** {avg_profit:.2f} / {avg_loss:.2f}"
                 })

            add_metric_field("Expectancy", 'expectancy')
            add_metric_field("Final Equity", 'final_equity')


            embed = {
                "title": f"Backtest Results: {strategy_name}", # Removed emoji
                "description": f"**Symbols:** `{symbols}`\n**Period:** {start_date} to {end_date} ({duration} days)",
                "color": 0x4CAF50,  # Green color
                "timestamp": datetime.utcnow().isoformat(),
                "fields": fields_list, # Use the generated list
                "footer": {
                    "text": "ZT-3 Backtester"
                }
            }

            payload = {
                "embeds": [embed]
            }

            response = requests.post(self.webhooks['backtest_results'], json=payload)

            if response.status_code >= 200 and response.status_code < 300:
                logger.info("Backtest results successfully sent to Discord.")
                return True
            else:
                logger.error(f"Failed to send Discord notification. Status: {response.status_code}, Response: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}", exc_info=True)
            return False

    def send_error_notification(self, title: str, error_msg: str) -> bool:
        """
        Send error notification.
        
        Args:
            title: Error title
            error_msg: Error message
            
        Returns:
            True if notification was sent, False otherwise
        """
        return self.send_system_notification(title, error_msg, "error")