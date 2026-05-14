"""
GoldBot Risk Manager
Dollar-based limits untuk akun $100.
Include tracking komisi MIFX $5/lot.
"""

import json
import os
import logging
from datetime import datetime
from typing import Tuple


class RiskManager:

    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.state  = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.config.state_file):
            try:
                with open(self.config.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'starting_balance': None,
            'daily_pnl':        0.0,
            'daily_trades':     0,
            'total_loss':       0.0,
            'last_date':        None,
            'halted':           False,
            'halt_reason':      None,
        }

    def _save(self):
        try:
            with open(self.config.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"State save error: {e}")

    def initialize(self, balance: float):
        if self.state['starting_balance'] is None:
            self.state['starting_balance'] = balance
            self.logger.info(f"Starting balance: ${balance:.2f}")

        today = datetime.now().strftime('%Y-%m-%d')
        if self.state['last_date'] != today:
            self.state['daily_pnl']    = 0.0
            self.state['daily_trades'] = 0
            self.state['last_date']    = today
            self.logger.info(f"New day reset | Balance: ${balance:.2f}")
        self._save()

    def check_can_trade(self, balance: float) -> Tuple[bool, str]:
        if self.state['halted']:
            return False, f"HALTED: {self.state['halt_reason']}"

        # Total loss check
        if self.state['starting_balance']:
            total_loss = self.state['starting_balance'] - balance
            if total_loss >= self.config.max_total_loss:
                self.state['halted']      = True
                self.state['halt_reason'] = f"Total loss ${total_loss:.2f} >= ${self.config.max_total_loss}"
                self._save()
                self.logger.critical(f"HALT: {self.state['halt_reason']}")
                return False, self.state['halt_reason']

        # Daily loss
        if self.state['daily_pnl'] <= -self.config.max_daily_loss:
            return False, f"Daily loss limit: ${self.state['daily_pnl']:.2f}"

        # Daily profit target
        if self.state['daily_pnl'] >= self.config.daily_profit_target:
            return False, f"Daily target reached: ${self.state['daily_pnl']:.2f} — stop for today!"

        # Max trades
        if self.state['daily_trades'] >= self.config.max_daily_trades:
            return False, f"Max {self.config.max_daily_trades} trades/day reached"

        return True, "OK"

    def record_trade(self, pnl_gross: float):
        """
        pnl_gross = profit/loss dari MT5 (sudah include komisi broker otomatis)
        """
        self.state['daily_pnl']    += pnl_gross
        self.state['daily_trades'] += 1
        if pnl_gross < 0:
            self.state['total_loss'] += abs(pnl_gross)
        self._save()

        status = "WIN" if pnl_gross > 0 else "LOSS"
        self.logger.info(
            f"Trade {status}: ${pnl_gross:+.2f} | "
            f"Daily: ${self.state['daily_pnl']:+.2f} | "
            f"Trades: {self.state['daily_trades']}/{self.config.max_daily_trades}"
        )

    def get_daily_summary(self) -> str:
        return (
            f"Daily P&L: ${self.state['daily_pnl']:+.2f} | "
            f"Trades: {self.state['daily_trades']} | "
            f"Remaining room: ${self.config.max_daily_loss + self.state['daily_pnl']:.2f}"
        )

    def reset_halt(self):
        self.state['halted']      = False
        self.state['halt_reason'] = None
        self._save()
        self.logger.warning("Halt manually reset")

    @property
    def daily_pnl(self) -> float:
        return self.state.get('daily_pnl', 0.0)

    @property
    def daily_trades(self) -> int:
        return self.state.get('daily_trades', 0)
