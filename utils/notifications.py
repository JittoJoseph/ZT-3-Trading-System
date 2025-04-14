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
        
        # Discord configuration - updated to support multiple webhooks
        self.discord_enabled = self.notifications_config.get('discord', {}).get('enabled', False)
        
        # Get webhook URLs for different notification types
        self.webhooks = {}
        discord_config = self.notifications_config.get('discord', {})
        
        if self.discord_enabled and 'webhooks' in discord_config:
            self.webhooks = discord_config.get('webhooks', {})
            
            # Check if we have at least one valid webhook
            if not any(self.webhooks.values()):
                logger.warning("Discord notifications are enabled but no webhook URLs are configured")
                self.discord_enabled = False
        else:
            # Fallback to single webhook URL for backwards compatibility
            single_webhook = discord_config.get('webhook_url', '')
            if (single_webhook):
                # Use the same URL for all notification types
                self.webhooks = {
                    'trade_alerts': single_webhook,
                    'performance': single_webhook,
                    'signals': single_webhook,
                    'system_status': single_webhook,
                    'backtest_results': single_webhook  # Add backtest webhook to default configuration
                }
            else:
                logger.warning("Discord notifications are enabled but webhook URLs are not configured")
                self.discord_enabled = False
        
        # Get notification levels
        self.notification_levels = self.notifications_config.get('notification_levels', {
            'trade_alerts': True,
            'performance': True,
            'signals': True,
            'system_status': True,
            'backtest_results': True  # Add backtest notification level
        })
            
        logger.info(f"Notification manager initialized. Discord enabled: {self.discord_enabled}")
        if self.discord_enabled:
            configured_channels = [k for k, v in self.webhooks.items() if v]
            logger.info(f"Configured Discord channels: {', '.join(configured_channels)}")
    
    def send_signal_notification(self, signal: Signal) -> bool:
        """
        Send a notification about a trading signal.
        
        Args:
            signal: The trading signal to send notification about
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled or not self.notification_levels.get('signals', True):
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
        
        webhook_url = self.webhooks.get('signals')
        if not webhook_url:
            logger.warning("Signal notification not sent: No webhook URL configured for signals")
            return False
            
        return self._send_discord_message(webhook_url, embeds=[embed])
    
    def send_trade_execution_notification(self, trade_details: Dict[str, Any]) -> bool:
        """
        Send a notification about a trade execution.
        
        Args:
            trade_details: Dictionary with trade execution details
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled or not self.notification_levels.get('trade_alerts', True):
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
        
        webhook_url = self.webhooks.get('trade_alerts')
        if not webhook_url:
            logger.warning("Trade notification not sent: No webhook URL configured for trade alerts")
            return False
            
        return self._send_discord_message(webhook_url, embeds=[embed])
    
    def send_error_notification(self, error_message: str, error_details: Optional[str] = None) -> bool:
        """
        Send a notification about an error.
        
        Args:
            error_message: Main error message
            error_details: Optional additional details
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled or not self.notification_levels.get('system_status', True):
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
        
        webhook_url = self.webhooks.get('system_status')
        if not webhook_url:
            logger.warning("Error notification not sent: No webhook URL configured for system status")
            return False
            
        return self._send_discord_message(webhook_url, embeds=[embed])
    
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
        if not self.discord_enabled or not self.notification_levels.get('system_status', True):
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
        
        webhook_url = self.webhooks.get('system_status')
        if not webhook_url:
            logger.warning("System notification not sent: No webhook URL configured for system status")
            return False
            
        return self._send_discord_message(webhook_url, embeds=[embed])
    
    def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Send a daily trading summary.
        
        Args:
            summary: Dictionary with daily summary data
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled or not self.notification_levels.get('performance', True):
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
            "color": 7506394 if pnl >= 0 else 16711680,  # Green if profitable, red if not
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
                    "value": f"{worst_symbol}: -₹{abs(worst_pnl)::.2f}",
                    "inline": True
                })
        
        webhook_url = self.webhooks.get('performance')
        if not webhook_url:
            logger.warning("Performance notification not sent: No webhook URL configured for performance")
            return False
            
        return self._send_discord_message(webhook_url, embeds=[embed])
    
    def send_backtest_results(self, results: Dict[str, Any]) -> bool:
        """
        Send backtest results as a detailed Discord embed.
        
        Args:
            results: Dictionary with backtest results data
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.discord_enabled or not self.notification_levels.get('backtest_results', True):
            return False
            
        # Extract key backtest data
        strategy_name = results.get('strategy_name', 'Unknown Strategy')
        symbol = results.get('symbol', 'Unknown')
        start_date = results.get('start_date', 'Unknown')
        end_date = results.get('end_date', 'Unknown')
        
        # Performance metrics
        total_pnl = results.get('net_pnl', 0.0)
        pnl_percent = results.get('net_pnl_percent', 0.0)
        total_trades = results.get('total_trades', 0)
        win_trades = results.get('win_trades', 0)
        win_rate = results.get('win_rate', 0.0)
        max_drawdown = results.get('max_drawdown_percent', 0.0)
        profit_factor = results.get('profit_factor', 0.0)
        sharpe_ratio = results.get('sharpe_ratio', 0.0)
        
        # Create title and color based on performance
        title = f"📊 Backtest Results: {strategy_name} on {symbol}"
        color = 5763719 if total_pnl >= 0 else 15158332  # Green if profitable, red if not
        
        # Create description with basic overview
        description = f"**Period**: {start_date} to {end_date}\n"
        description += f"**Net P&L**: {'+' if total_pnl >= 0 else ''}₹{total_pnl:.2f} ({pnl_percent:.2f}%)\n"
        description += f"**Win Rate**: {win_rate:.2f}% ({win_trades}/{total_trades} trades)\n"
        description += f"**Max Drawdown**: {max_drawdown:.2f}%"
        
        # Create embed structure
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": f"ZT-3 Backtesting System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            },
            "fields": [
                {
                    "name": "Performance Metrics",
                    "value": (
                        f"Profit Factor: {profit_factor:.2f}\n"
                        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                        f"Total Trades: {total_trades}\n"
                        f"Win/Loss: {win_trades}/{total_trades-win_trades}\n"
                    ),
                    "inline": True
                }
            ]
        }
        
        # Add trade statistics if available
        if 'avg_win' in results and 'avg_loss' in results:
            avg_win = results.get('avg_win', 0.0)
            avg_loss = results.get('avg_loss', 0.0)
            avg_trade = results.get('avg_trade', 0.0)
            max_win = results.get('max_win', 0.0)
            max_loss = results.get('max_loss', 0.0)
            
            embed["fields"].append({
                "name": "Trade Statistics",
                "value": (
                    f"Avg Win: ₹{avg_win:.2f}\n"
                    f"Avg Loss: ₹{avg_loss:.2f}\n"
                    f"Avg Trade: ₹{avg_trade:.2f}\n"
                    f"Max Win: ₹{max_win:.2f}\n"
                    f"Max Loss: ₹{max_loss:.2f}\n"
                ),
                "inline": True
            })
        
        # Add parameter information if available
        if 'parameters' in results:
            params = results.get('parameters', {})
            param_text = ""
            for key, value in params.items():
                param_text += f"{key}: {value}\n"
            
            if param_text:
                embed["fields"].append({
                    "name": "Strategy Parameters",
                    "value": param_text,
                    "inline": False
                })
        
        # Add monthly returns if available
        if 'monthly_returns' in results:
            monthly_returns = results.get('monthly_returns', {})
            if monthly_returns:
                # Format monthly returns (limit to recent months to avoid too long text)
                months = list(monthly_returns.keys())[-6:]  # Last 6 months
                monthly_text = ""
                for month in months:
                    monthly_return = monthly_returns[month]
                    sign = '+' if monthly_return >= 0 else ''
                    monthly_text += f"{month}: {sign}{monthly_return:.2f}%\n"
                
                embed["fields"].append({
                    "name": "Recent Monthly Returns",
                    "value": monthly_text or "No monthly data",
                    "inline": False
                })
        
        # Check for dedicated backtest webhook URL
        webhook_url = self.webhooks.get('backtest_results')
        if not webhook_url:
            logger.warning("Backtest results notification not sent: No webhook URL configured for backtest_results")
            return False
            
        # Send the embed to Discord
        return self._send_discord_message(webhook_url, embeds=[embed])
    
    def _send_discord_message(self, webhook_url: str, content: Optional[str] = None, embeds: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Send a message to Discord using webhook.
        
        Args:
            webhook_url: Discord webhook URL
            content: Text content of the message
            embeds: List of Discord embeds
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.discord_enabled or not webhook_url:
            return False
            
        payload = {}
        
        if content:
            payload["content"] = content
            
        if embeds:
            payload["embeds"] = embeds
            
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                webhook_url, 
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