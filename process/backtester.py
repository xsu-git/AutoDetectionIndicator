#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/9 11:14     Xsu         1.0         None
'''
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型枚举"""
    STRONG_BUY = "STRONG_BUY"
    MODERATE_BUY = "MODERATE_BUY"
    WEAK_BUY = "WEAK_BUY"
    WATCH_BUY = "WATCH_BUY"
    STRONG_SELL = "STRONG_SELL"
    MODERATE_SELL = "MODERATE_SELL"
    WEAK_SELL = "WEAK_SELL"
    WATCH_SELL = "WATCH_SELL"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class Position:
    """持仓信息"""
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    direction: int  # 1 for long, -1 for short
    stop_loss: float
    take_profit: float
    signal_type: str


class EnhancedBacktester:
    """
    增强版回测系统
    - 支持多层信号处理
    - 动态仓位管理
    - 追踪止损功能
    - 详细性能分析
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_positions: int = 5):
        """
        初始化回测器

        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点
            max_positions: 最大同时持仓数
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        self.positions: List[Position] = []
        self.closed_trades = []

    def backtest(self, signals_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
        """
        执行回测

        Args:
            signals_df: 信号数据
            price_df: 价格数据

        Returns:
            回测结果字典
        """
        results = {
            'trades': [],
            'equity_curve': [],
            'positions': [],
            'daily_returns': []
        }

        capital = self.initial_capital
        available_capital = capital

        for idx, row in signals_df.iterrows():
            if idx not in price_df.index:
                continue

            current_price = price_df.loc[idx, 'close']

            # 1. 检查现有持仓的止损止盈
            self._check_exits(idx, current_price, results, available_capital)

            # 2. 处理新信号
            signal = row.get('trading_signal', 'NO_SIGNAL')

            if signal != 'NO_SIGNAL' and len(self.positions) < self.max_positions:
                position_size = row.get('position_size', 0.1)

                if 'STRONG_BUY' in signal:
                    self._open_long_position(
                        idx, current_price, position_size,
                        available_capital, signal, row, results
                    )
                elif 'STRONG_SELL' in signal:
                    self._open_short_position(
                        idx, current_price, position_size,
                        available_capital, signal, row, results
                    )

            # 3. 更新追踪止损
            self._update_trailing_stops(current_price)

            # 4. 计算当前权益
            position_value = sum([
                pos.size * pos.direction * (current_price - pos.entry_price)
                for pos in self.positions
            ])

            total_value = available_capital + position_value

            results['equity_curve'].append({
                'timestamp': idx,
                'capital': available_capital,
                'position_value': position_value,
                'total_value': total_value,
                'num_positions': len(self.positions)
            })

            # 5. 计算日收益率
            if len(results['equity_curve']) > 1:
                prev_value = results['equity_curve'][-2]['total_value']
                daily_return = (total_value - prev_value) / prev_value
                results['daily_returns'].append(daily_return)

        # 计算最终指标
        results['metrics'] = self._calculate_comprehensive_metrics(results)

        return results

    def _open_long_position(self, timestamp, price, size_pct, available_capital,
                            signal, row, results):
        """开多仓"""
        position_capital = available_capital * size_pct
        entry_price = price * (1 + self.slippage)  # 考虑滑点
        commission = position_capital * self.commission

        # 计算实际仓位大小
        actual_size = (position_capital - commission) / entry_price

        if actual_size > 0:
            position = Position(
                entry_time=timestamp,
                entry_price=entry_price,
                size=actual_size,
                direction=1,
                stop_loss=row.get('stop_loss_long', entry_price * 0.98),
                take_profit=row.get('take_profit_long', entry_price * 1.02),
                signal_type=signal
            )

            self.positions.append(position)

            results['trades'].append({
                'timestamp': timestamp,
                'type': 'OPEN_LONG',
                'signal': signal,
                'price': entry_price,
                'size': actual_size,
                'commission': commission
            })

    def _open_short_position(self, timestamp, price, size_pct, available_capital,
                             signal, row, results):
        """开空仓"""
        position_capital = available_capital * size_pct
        entry_price = price * (1 - self.slippage)  # 考虑滑点
        commission = position_capital * self.commission

        actual_size = (position_capital - commission) / entry_price

        if actual_size > 0:
            position = Position(
                entry_time=timestamp,
                entry_price=entry_price,
                size=actual_size,
                direction=-1,
                stop_loss=row.get('stop_loss_short', entry_price * 1.02),
                take_profit=row.get('take_profit_short', entry_price * 0.98),
                signal_type=signal
            )

            self.positions.append(position)

            results['trades'].append({
                'timestamp': timestamp,
                'type': 'OPEN_SHORT',
                'signal': signal,
                'price': entry_price,
                'size': actual_size,
                'commission': commission
            })

    def _check_exits(self, timestamp, current_price, results, available_capital):
        """检查止损止盈"""
        positions_to_close = []

        for i, pos in enumerate(self.positions):
            should_close = False
            exit_reason = ""

            if pos.direction == 1:  # 多仓
                if current_price <= pos.stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price >= pos.take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
            else:  # 空仓
                if current_price >= pos.stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price <= pos.take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"

            if should_close:
                positions_to_close.append((i, exit_reason))

        # 平仓
        for idx, reason in reversed(positions_to_close):
            pos = self.positions.pop(idx)
            exit_price = current_price * (1 - self.slippage * pos.direction)

            pnl = pos.size * pos.direction * (exit_price - pos.entry_price)
            commission = pos.size * exit_price * self.commission
            net_pnl = pnl - commission

            results['trades'].append({
                'timestamp': timestamp,
                'type': f'CLOSE_{reason}',
                'signal': pos.signal_type,
                'entry_price': pos.entry_price,
                'exit_price': exit_price,
                'size': pos.size,
                'pnl': net_pnl,
                'return': net_pnl / (pos.size * pos.entry_price),
                'holding_period': (timestamp - pos.entry_time).total_seconds() / 3600  # 小时
            })

            self.closed_trades.append(results['trades'][-1])

    def _update_trailing_stops(self, current_price):
        """更新追踪止损"""
        for pos in self.positions:
            if pos.signal_type.startswith('STRONG'):
                # 强信号使用追踪止损
                trailing_pct = 0.015  # 1.5%追踪止损

                if pos.direction == 1:  # 多仓
                    new_stop = current_price * (1 - trailing_pct)
                    pos.stop_loss = max(pos.stop_loss, new_stop)
                else:  # 空仓
                    new_stop = current_price * (1 + trailing_pct)
                    pos.stop_loss = min(pos.stop_loss, new_stop)

    def _calculate_comprehensive_metrics(self, results) -> Dict:
        """计算综合回测指标"""

        if not results['equity_curve']:
            return {}

        equity_df = pd.DataFrame(results['equity_curve'])
        trades_df = pd.DataFrame(results['trades'])

        # 基础收益指标
        total_return = (equity_df['total_value'].iloc[-1] / self.initial_capital - 1)

        # 交易统计
        closed_trades = [t for t in results['trades'] if 'pnl' in t]
        total_trades = len(closed_trades)

        if total_trades > 0:
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]

            win_rate = len(winning_trades) / total_trades
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum([t['pnl'] for t in winning_trades]) /
                                sum([t['pnl'] for t in losing_trades])) if losing_trades else np.inf

            # 按信号类型统计
            signal_stats = self._calculate_signal_statistics(closed_trades)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            signal_stats = {}

        # 风险指标
        if results['daily_returns']:
            returns = np.array(results['daily_returns'])
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Calmar比率
            calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            sharpe = 0
            max_drawdown = 0
            calmar = 0

        return {
            # 收益指标
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365 / len(equity_df)) - 1 if len(equity_df) > 0 else 0,

            # 交易统计
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            # 风险指标
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,

            # 信号统计
            'signal_statistics': signal_stats,

            # 执行统计
            'avg_holding_period': np.mean([t.get('holding_period', 0) for t in closed_trades]) if closed_trades else 0,
            'max_consecutive_wins': self._calculate_max_consecutive(winning_trades),
            'max_consecutive_losses': self._calculate_max_consecutive(losing_trades)
        }

    def _calculate_signal_statistics(self, trades: List[Dict]) -> Dict:
        """按信号类型统计性能"""
        signal_stats = {}

        # 按信号类型分组
        signal_groups = {}
        for trade in trades:
            signal = trade.get('signal', 'UNKNOWN')
            if signal not in signal_groups:
                signal_groups[signal] = []
            signal_groups[signal].append(trade)

        # 计算每种信号的统计
        for signal, group_trades in signal_groups.items():
            if group_trades:
                wins = [t for t in group_trades if t.get('pnl', 0) > 0]
                signal_stats[signal] = {
                    'count': len(group_trades),
                    'win_rate': len(wins) / len(group_trades),
                    'avg_return': np.mean([t.get('return', 0) for t in group_trades]),
                    'total_pnl': sum([t.get('pnl', 0) for t in group_trades])
                }

        return signal_stats

    def _calculate_max_consecutive(self, trades: List) -> int:
        """计算最大连续次数"""
        if not trades:
            return 0

        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(trades)):
            if trades[i].get('timestamp', 0) > trades[i - 1].get('timestamp', 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive