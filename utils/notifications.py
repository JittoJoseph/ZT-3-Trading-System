"""
Notifications Module for ZT-3 Trading System.

This module provides functionality for sending notifications
to Discord and other platforms about trading signals and system events.
"""

import logging
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import traceback

from strategy import Signal

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Notification manager for sending alerts to various channels.
    
    Supports Discord and can be extended to support other platforms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification manager.
        
        Args:
            config: Configuration dictionary containing notification settings
        """
        self.config = config
        self.notifications_config = config.get('notifications', {})
        
        # Discord configuration
        self.discord_enabled = self.notifications_config.get('discord', {}).get('enabled', False)
        self.discord_webhook_url = self.notifications_config.get('discord', {}).get('webhook_url', '')
        
        if self.discord_enabled and not self.discord_webhook_url:
            logger.warning("Discord notifications are enabled but webhook URL is not configured")
            self.discord_enabled = False
            
        logger.info(f"Notification manager initialized. Discord enabled: {self.discord_enabled}")
    
    def send_signal_notification(self, signal: Signal) -> bool:
        """
        Send a notification about a trading signal.
        
        Args:
            signal: The trading signal to send notification about
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled:
            return False
            
        # Create signal message
        signal_type = signal.signal_type.value
        symbol = signal.symbol
        price = signal.price
        
        if signal_type == "ENTRY":
            title = f"🔵 ENTRY SIGNAL: {symbol}"
            color = 3447003  # Discord Blue
            description = f"Entry signal generated for {symbol} at price ₹{price:.2f}"
            
            # Add take profit and stop loss if available
            if signal.take_profit_level:
                description += f"\nTake Profit: ₹{signal.take_profit_level:.2f}"
            if signal.stop_loss_level:
                description += f"\nStop Loss: ₹{signal.stop_loss_level:.2f}"
                
        else:  # EXIT
            title = f"🔴 EXIT SIGNAL: {symbol}"
            color = 15158332  # Discord Red
            description = f"Exit signal generated for {symbol} at price ₹{price:.2f}"
            
            # Add exit reason if available
            if signal.exit_reason:
                description += f"\nExit Reason: {signal.exit_reason.value}"
        
        # Create and send Discord embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            },
            "fields": [
                {
                    "name": "Symbol",
                    "value": symbol,
                    "inline": True
                },
                {
                    "name": "Price",
                    "value": f"₹{price:.2f}",
                    "inline": True
                },
                {
                    "name": "Time",
                    "value": signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "inline": True
                }
            ]
        }
        
        return self._send_discord_message(embeds=[embed])
    
    def send_trade_execution_notification(self, trade_details: Dict[str, Any]) -> bool:
        """
        Send a notification about a trade execution.
        
        Args:
            trade_details: Dictionary with trade execution details
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled:
            return False
        
        # Extract trade details
        symbol = trade_details.get('symbol', 'Unknown')
        action = trade_details.get('action', 'Unknown')
        price = float(trade_details.get('price', 0.0))
        quantity = int(trade_details.get('quantity', 0))
        
        if action == "BUY":
            title = f"🟢 BUY EXECUTED: {symbol}"
            color = 5763719  # Green
            description = f"Buy order executed for {quantity} shares of {symbol} at price ₹{price:.2f}"
            
        else:  # SELL
            title = f"🟣 SELL EXECUTED: {symbol}"
            color = 15105570  # Purple
            description = f"Sell order executed for {quantity} shares of {symbol} at price ₹{price:.2f}"
            
            # Add P&L if available
            if 'pnl' in trade_details:
                pnl = float(trade_details['pnl'])
                pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"-₹{abs(pnl):.2f}"
                description += f"\nP&L: {pnl_str}"
                
            # Add exit reason if available
            if 'exit_reason' in trade_details:
                description += f"\nExit Reason: {trade_details['exit_reason']}"
        
        # Create and send Discord embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            },
            "fields": [
                {
                    "name": "Symbol",
                    "value": symbol,
                    "inline": True
                },
                {
                    "name": "Price",
                    "value": f"₹{price:.2f}",
                    "inline": True
                },
                {
                    "name": "Quantity",
                    "value": str(quantity),
                    "inline": True
                }
            ]
        }
        
        # Add commission information if available
        if 'commission' in trade_details:
            embed["fields"].append({
                "name": "Commission",
                "value": f"₹{float(trade_details['commission']):.2f}",
                "inline": True
            })
        
        return self._send_discord_message(embeds=[embed])
    
    def send_error_notification(self, error_message: str, error_details: Optional[str] = None) -> bool:
        """
        Send a notification about an error.
        
        Args:
            error_message: Main error message
            error_details: Optional additional details
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled:
            return False
            
        title = "⚠️ SYSTEM ERROR"
        description = error_message
        
        # Create and send Discord embed
        embed = {
            "title": title,
            "description": description,
            "color": 16711680,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        # Add error details if available
        if error_details:
            embed["fields"] = [
                {
                    "name": "Error Details",
                    "value": error_details[:1000] + "..." if len(error_details) > 1000 else error_details,
                }
            ]
        
        return self._send_discord_message(embeds=[embed])
    
    def send_system_notification(self, title: str, message: str, level: str = "info") -> bool:
        """
        Send a system notification.
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level (info, warning, error)
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled:
            return False
            
        # Set color based on level
        if level.lower() == "warning":
            color = 16763904  # Yellow
            title = f"⚠️ {title}"
        elif level.lower() == "error":
            color = 16711680  # Red
            title = f"🚨 {title}"
        else:  # Info
            color = 3447003  # Blue
            title = f"ℹ️ {title}"
        
        # Create and send Discord embed
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        return self._send_discord_message(embeds=[embed])
    
    def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Send a daily trading summary.
        
        Args:
            summary: Dictionary with daily summary data
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled:
            return False
            
        # Extract summary data
        date = summary.get('date', datetime.now().strftime("%Y-%m-%d"))
        pnl = summary.get('daily_pnl', 0.0)
        total_trades = summary.get('total_trades', 0)
        win_trades = summary.get('win_trades', 0)
        loss_trades = summary.get('loss_trades', 0)
        
        # Calculate win rate
        win_rate = 0.0
        if total_trades > 0:
            win_rate = (win_trades / total_trades) * 100
            
        # Create title and description
        title = f"📊 Daily Summary: {date}"
        
        # Format P&L with color indicator
        if pnl >= 0:
            pnl_str = f"📈 +₹{pnl:.2f}"
        else:
            pnl_str = f"📉 -₹{abs(pnl):.2f}"
            
        description = f"**Daily P&L: {pnl_str}**\n\n"
        description += f"Total Trades: {total_trades}\n"
        description += f"Win Rate: {win_rate:.1f}% ({win_trades}W / {loss_trades}L)"
        
        # Create and send Discord embed
        embed = {
            "title": title,
            "description": description,
            "color": 7506394,  # Green if profitable, red if not
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "ZT-3 Trading System"
            }
        }
        
        # Add best and worst trades if available
        if 'best_trade' in summary and 'symbol' in summary['best_trade']:
            best_trade = summary['best_trade']
            best_symbol = best_trade.get('symbol', '')
            best_pnl = best_trade.get('pnl', 0.0)
            
            if best_symbol and best_pnl > 0:
                embed["fields"] = embed.get("fields", [])
                embed["fields"].append({
                    "name": "Best Trade",
                    "value": f"{best_symbol}: +₹{best_pnl:.2f}",
                    "inline": True
                })
                
        if 'worst_trade' in summary and 'symbol' in summary['worst_trade']:
            worst_trade = summary['worst_trade']
            worst_symbol = worst_trade.get('symbol', '')
            worst_pnl = worst_trade.get('pnl', 0.0)
            
            if worst_symbol and worst_pnl < 0:
                embed["fields"] = embed.get("fields", [])
                embed["fields"].append({
                    "name": "Worst Trade",
                    "value": f"{worst_symbol}: -₹{abs(worst_pnl):.2f}",
                    "inline": True
                })
        
        return self._send_discord_message(embeds=[embed])
    
    def _send_discord_message(self, content: Optional[str] = None, embeds: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Send a message to Discord using webhook.
        
        Args:
            content: Text content of the message
            embeds: List of Discord embeds
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.discord_enabled or not self.discord_webhook_url:
            return False
            
        payload = {}
        
        if content:
            payload["content"] = content
            
        if embeds:
            payload["embeds"] = embeds
            
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.discord_webhook_url, 
                data=json.dumps(payload), 
                headers=headers
            )
            response.raise_for_status()
            
            logger.debug(f"Discord notification sent successfully. Status code: {response.status_code}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {str(e)}")
            logger.debug(traceback.format_exc())
            return False