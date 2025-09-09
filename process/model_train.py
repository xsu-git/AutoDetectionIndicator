#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 16:49     Xsu         1.0         None
'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np
from utils import logBot

class MachineLearnTrain:

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False

    def train(self, X: pd.DataFrame, y_dict: dict, feature_cols: list):
        """训练多个预测模型"""
        print("🚀 开始训练机器学习模型...")

        # 数据预处理
        scaler = RobustScaler()  # 对异常值更鲁棒
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X.fillna(0)),
            columns=X.columns,
            index=X.index
        )
        self.scalers['features'] = scaler

        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)

        # 训练不同类型的模型
        self.models = {}

        # 1. 回归模型 - 预测收益率
        print("📈 训练收益率回归模型...")
        for target in ['future_return_3', 'future_return_5']:
            if target in y_dict:
                y_clean = y_dict[target].fillna(0)

                # 梯度提升回归
                gb_regressor = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    subsample=0.8
                )

                # 交叉验证评分
                cv_scores = cross_val_score(gb_regressor, X_scaled, y_clean, cv=tscv, scoring='neg_mean_squared_error')
                print(f"  {target} - CV MSE: {-cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

                # 训练最终模型
                gb_regressor.fit(X_scaled, y_clean)
                self.models[target] = gb_regressor

        # 2. 分类模型 - 预测方向和显著性
        print("🎯 训练分类模型...")
        for target in ['direction_3', 'direction_5', 'significant_move_3', 'significant_move_5']:
            if target in y_dict:
                y_clean = y_dict[target].fillna(0)

                # 随机森林分类器
                rf_classifier = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced'
                )

                # 交叉验证
                cv_scores = cross_val_score(rf_classifier, X_scaled, y_clean, cv=tscv, scoring='accuracy')
                print(f"  {target} - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                # 训练最终模型
                rf_classifier.fit(X_scaled, y_clean)
                self.models[target] = rf_classifier

        self.is_trained = True
        self.feature_cols = feature_cols
        print("✅ 模型训练完成!")

        return self.models

    def generate_trading_signals(self, df: pd.DataFrame, features_df:pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train_models()")

        print("🔮 生成交易信号...")

        # 提取特征
        X = features_df[self.feature_cols].copy()

        # 数据预处理
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X.fillna(0)),
            columns=X.columns,
            index=features_df.index
        )



        # 生成预测
        signals_df = pd.DataFrame(index=df.index)
        signals_df['timestamp'] = df.index
        signals_df['close_price'] = df['close']

        idx = X_scaled.index

        # 生成预测
        for target in self.models.keys():
            if 'future_return' in target:
                pred = pd.Series(self.models[target].predict(X_scaled), index=idx)
                signals_df.loc[idx, f'predicted_{target}'] = pred
            else:
                pred = pd.Series(self.models[target].predict(X_scaled), index=idx)
                signals_df.loc[idx, f'predicted_{target}'] = pred

                if hasattr(self.models[target], "predict_proba"):
                    proba = self.models[target].predict_proba(X_scaled)
                    max_prob = pd.Series(np.max(proba, axis=1), index=idx)
                    signals_df.loc[idx, f'{target}_confidence'] = max_prob

        # 使用增强版综合信号生成
        signals_df = self._generate_flexible_composite_signals(signals_df, df)

        return signals_df



    def _generate_flexible_composite_signals(self, signals_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        灵活的综合信号生成系统
        使用区间判断而非固定阈值
        """

        # 1. 计算市场状态指标
        signals_df = self._calculate_market_regime(signals_df, df)

        # 2. 多维度信号强度评分
        signals_df['bull_score'] = 0.0
        signals_df['bear_score'] = 0.0

        # 收益率预测贡献（使用区间映射）
        if 'predicted_future_return_3' in signals_df.columns:
            # 将预测收益率映射到0-10分
            return_3 = signals_df['predicted_future_return_3']

            # 动态区间：根据历史分位数调整
            bull_threshold_low = return_3.quantile(0.55)  # 降低阈值
            bull_threshold_high = return_3.quantile(0.85)
            bear_threshold_high = return_3.quantile(0.45)  # 降低阈值
            bear_threshold_low = return_3.quantile(0.15)

            # 看涨评分
            bull_mask = return_3 > bull_threshold_low
            signals_df.loc[bull_mask, 'bull_score'] += np.interp(
                return_3[bull_mask],
                [bull_threshold_low, bull_threshold_high],
                [1, 5]
            )

            # 看跌评分
            bear_mask = return_3 < bear_threshold_high
            signals_df.loc[bear_mask, 'bear_score'] += np.interp(
                return_3[bear_mask],
                [bear_threshold_low, bear_threshold_high],
                [5, 1]
            )

        # 方向预测贡献（考虑置信度）
        if 'predicted_direction_3' in signals_df.columns and 'direction_3_confidence' in signals_df.columns:
            direction = signals_df['predicted_direction_3']
            confidence = signals_df['direction_3_confidence']

            # 使用置信度区间
            conf_low = 0.5  # 降低置信度要求
            conf_high = 0.8

            # 看涨信号
            bull_dir_mask = (direction == 1) & (confidence > conf_low)
            signals_df.loc[bull_dir_mask, 'bull_score'] += np.interp(
                confidence[bull_dir_mask],
                [conf_low, conf_high],
                [1, 4]
            )

            # 看跌信号
            bear_dir_mask = (direction == -1) & (confidence > conf_low)
            signals_df.loc[bear_dir_mask, 'bear_score'] += np.interp(
                confidence[bear_dir_mask],
                [conf_low, conf_high],
                [1, 4]
            )

        # 显著性移动预测贡献
        if 'significant_move_3_confidence' in signals_df.columns:
            sig_conf = signals_df['significant_move_3_confidence']

            # 显著移动加分（双向）
            sig_mask = sig_conf > 0.45  # 降低阈值
            bonus = np.interp(sig_conf[sig_mask], [0.45, 0.8], [0.5, 2])

            signals_df.loc[sig_mask, 'bull_score'] += bonus
            signals_df.loc[sig_mask, 'bear_score'] += bonus

        # 3. 技术指标确认加分
        signals_df = self._add_technical_confirmation(signals_df, df)

        # 4. 生成多层次交易信号
        signals_df = self._generate_tiered_signals(signals_df)

        # 5. 动态仓位计算
        signals_df['position_size'] = self._calculate_dynamic_position(signals_df, df)

        # 6. 自适应止损止盈
        signals_df = self._calculate_adaptive_stops(signals_df, df)

        return signals_df

    def _calculate_market_regime(self, signals_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """识别市场状态"""

        # 计算短期和长期趋势
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            signals_df['trend_state'] = np.where(
                df['sma_20'] > df['sma_50'], 1,  # 上升趋势
                np.where(df['sma_20'] < df['sma_50'], -1, 0)  # 下降趋势
            )

        # 计算波动率状态
        if 'atr_14' in df.columns:
            atr_percentile = df['atr_14'].rolling(50).rank(pct=True)
            signals_df['volatility_state'] = np.where(
                atr_percentile > 0.7, 'high',
                np.where(atr_percentile < 0.3, 'low', 'normal')
            )

        return signals_df

    def _add_technical_confirmation(self, signals_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标确认"""

        # RSI确认
        if 'rsi_14' in df.columns:
            # 超卖区间（更宽松）
            oversold_mask = (df['rsi_14'] < 35) & (df['rsi_14'] > 20)
            signals_df.loc[oversold_mask, 'bull_score'] += 1.5

            # 超买区间（更宽松）
            overbought_mask = (df['rsi_14'] > 65) & (df['rsi_14'] < 80)
            signals_df.loc[overbought_mask, 'bear_score'] += 1.5

        # MACD确认
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_bull = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            macd_bear = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

            signals_df.loc[macd_bull, 'bull_score'] += 2
            signals_df.loc[macd_bear, 'bear_score'] += 2

        # 布林带确认
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            # 接近下轨（买入机会）
            near_lower = (df['close'] - df['bb_lower']) / (df['bb_middle'] - df['bb_lower']) < 0.2
            signals_df.loc[near_lower, 'bull_score'] += 1

            # 接近上轨（卖出机会）
            near_upper = (df['bb_upper'] - df['close']) / (df['bb_upper'] - df['bb_middle']) < 0.2
            signals_df.loc[near_upper, 'bear_score'] += 1

        return signals_df

    def _generate_tiered_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """生成多层次交易信号"""

        # 计算净得分
        signals_df['net_score'] = signals_df['bull_score'] - signals_df['bear_score']
        signals_df['total_score'] = signals_df['bull_score'] + signals_df['bear_score']

        # 定义更灵活的信号等级
        def categorize_signal_flexible(row):
            net = row['net_score']
            total = row['total_score']

            # 强信号（降低阈值）
            if total >= 6:
                if net >= 3:
                    return "STRONG_BUY"
                elif net <= -3:
                    return "STRONG_SELL"

            # 中等信号（新增）
            if total >= 3.5:
                if net >= 1.5:
                    return "MODERATE_BUY"
                elif net <= -1.5:
                    return "MODERATE_SELL"

            # 弱信号（新增）
            if total >= 2:
                if net >= 0.5:
                    return "WEAK_BUY"
                elif net <= -0.5:
                    return "WEAK_SELL"

            # 观察信号（新增）
            if total >= 1:
                if net > 0:
                    return "WATCH_BUY"
                elif net < 0:
                    return "WATCH_SELL"

            return "NO_SIGNAL"

        signals_df['trading_signal'] = signals_df.apply(categorize_signal_flexible, axis=1)

        # 添加信号强度（0-10）
        signals_df['signal_strength'] = np.clip(signals_df['total_score'] / 2, 0, 10)

        return signals_df

    def _calculate_dynamic_position(self, signals_df: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """动态仓位计算"""

        base_position = pd.Series(0.0, index=signals_df.index)

        # 根据信号强度分配仓位
        position_map = {
            'STRONG_BUY': 0.4,  # 最大40%仓位
            'STRONG_SELL': 0.4,
            'MODERATE_BUY': 0.25,  # 中等25%仓位
            'MODERATE_SELL': 0.25,
            'WEAK_BUY': 0.15,  # 弱信号15%仓位
            'WEAK_SELL': 0.15,
            'WATCH_BUY': 0.08,  # 观察仓位8%
            'WATCH_SELL': 0.08
        }

        for signal, size in position_map.items():
            mask = signals_df['trading_signal'] == signal
            base_position[mask] = size

        # 根据市场波动率调整
        if 'volatility_state' in signals_df.columns:
            # 高波动降低仓位
            high_vol_mask = signals_df['volatility_state'] == 'high'
            base_position[high_vol_mask] *= 0.7

            # 低波动可以适当增加仓位
            low_vol_mask = signals_df['volatility_state'] == 'low'
            base_position[low_vol_mask] *= 1.2

        # 根据置信度微调
        if 'direction_3_confidence' in signals_df.columns:
            confidence = signals_df['direction_3_confidence'].fillna(0.5)
            base_position *= (0.5 + confidence)  # 0.5-1.5倍调整

        return np.clip(base_position, 0, 0.5)  # 最大50%仓位

    def _calculate_adaptive_stops(self, signals_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """自适应止损止盈计算"""

        current_price = signals_df['close_price']

        # 基于ATR的动态止损
        if 'atr_14' in df.columns:
            atr = df['atr_14']

            # 根据信号强度调整止损距离
            stop_multiplier = pd.Series(2.0, index=signals_df.index)

            strong_signals = signals_df['trading_signal'].str.contains('STRONG')
            moderate_signals = signals_df['trading_signal'].str.contains('MODERATE')
            weak_signals = signals_df['trading_signal'].str.contains('WEAK')

            stop_multiplier[strong_signals] = 1.5  # 强信号止损更紧
            stop_multiplier[moderate_signals] = 2.0  # 中等信号标准止损
            stop_multiplier[weak_signals] = 2.5  # 弱信号止损更宽

            # 止损价格
            signals_df['stop_loss_distance'] = atr * stop_multiplier
            signals_df['stop_loss_long'] = current_price - signals_df['stop_loss_distance']
            signals_df['stop_loss_short'] = current_price + signals_df['stop_loss_distance']

            # 止盈价格（风险回报比）
            reward_ratio = pd.Series(2.0, index=signals_df.index)
            reward_ratio[strong_signals] = 3.0  # 强信号目标更高
            reward_ratio[moderate_signals] = 2.0
            reward_ratio[weak_signals] = 1.5

            signals_df['take_profit_distance'] = signals_df['stop_loss_distance'] * reward_ratio
            signals_df['take_profit_long'] = current_price + signals_df['take_profit_distance']
            signals_df['take_profit_short'] = current_price - signals_df['take_profit_distance']

        else:
            # 默认百分比止损止盈
            signals_df['stop_loss_pct'] = 0.02  # 2%止损
            signals_df['take_profit_pct'] = 0.04  # 4%止盈

            signals_df['stop_loss_long'] = current_price * (1 - signals_df['stop_loss_pct'])
            signals_df['take_profit_long'] = current_price * (1 + signals_df['take_profit_pct'])
            signals_df['stop_loss_short'] = current_price * (1 + signals_df['stop_loss_pct'])
            signals_df['take_profit_short'] = current_price * (1 - signals_df['take_profit_pct'])

        return signals_df


    # def _calculate_position_size(self, signals_df: pd.DataFrame) -> pd.Series:
    #     """计算建议仓位大小 (0-1之间)"""
    #     base_size = signals_df['signal_strength'] / 10 * 0.3  # 基础仓位最大30%
    #
    #     # 信号置信度调整
    #     confidence_avg = (
    #                              signals_df.get('direction_3_confidence', 0.5) +
    #                              signals_df.get('significant_move_3_confidence', 0.5)
    #                      ) / 2
    #
    #     adjusted_size = base_size * confidence_avg
    #
    #     return np.clip(adjusted_size, 0, 0.5)  # 最大50%仓位
    #
    # def _calculate_stop_take_levels(self, signals_df: pd.DataFrame) -> pd.DataFrame:
    #     """计算止损止盈水平"""
    #
    #     # 基于预期收益率的止盈
    #     expected_return_3 = signals_df.get('predicted_future_return_3', 0)
    #     expected_return_5 = signals_df.get('predicted_future_return_5', 0)
    #
    #     # 动态止盈 (预期收益的80%)
    #     signals_df['take_profit_pct'] = np.clip(
    #         np.maximum(abs(expected_return_3), abs(expected_return_5)) * 0.8,
    #         0.005,  # 最小0.5%
    #         0.05  # 最大5%
    #     )
    #
    #     # 动态止损 (预期收益的40%，但最大2%)
    #     signals_df['stop_loss_pct'] = np.clip(
    #         signals_df['take_profit_pct'] * 0.4,
    #         0.003,  # 最小0.3%
    #         0.02  # 最大2%
    #     )
    #
    #     # 计算具体价位
    #     current_price = signals_df['close_price']
    #
    #     signals_df['take_profit_long'] = current_price * (1 + signals_df['take_profit_pct'])
    #     signals_df['stop_loss_long'] = current_price * (1 - signals_df['stop_loss_pct'])
    #
    #     signals_df['take_profit_short'] = current_price * (1 - signals_df['take_profit_pct'])
    #     signals_df['stop_loss_short'] = current_price * (1 + signals_df['stop_loss_pct'])
    #
    #     return signals_df

    # def analyze_feature_importance(self) -> dict:
    #     """分析特征重要性"""
    #     if not self.is_trained:
    #         return {}
    #
    #     importance_dict = {}
    #
    #     for model_name, model in self.models.items():
    #         if hasattr(model, 'feature_importances_'):
    #             # 获取特征重要性
    #             importances = model.feature_importances_
    #             feature_importance = pd.DataFrame({
    #                 'feature': self.feature_cols,
    #                 'importance': importances
    #             }).sort_values('importance', ascending=False)
    #
    #             importance_dict[model_name] = feature_importance.head(20)
    #
    #             print(f"\n🔍 {model_name} - Top 10 重要特征:")
    #             for idx, row in feature_importance.head(10).iterrows():
    #                 print(f"  {row['feature']}: {row['importance']:.4f}")
    #
    #     return importance_dict

    # def backtest_signals(self, signals_df: pd.DataFrame) -> dict:
    #     """回测交易信号"""
    #     print("📊 执行信号回测...")
    #
    #     # 简单回测逻辑
    #     results = {
    #         'total_signals': 0,
    #         'buy_signals': 0,
    #         'sell_signals': 0,
    #         'strong_signals': 0,
    #         'win_rate': 0.0,
    #         'avg_return_per_signal': 0.0
    #     }
    #
    #     # 统计信号分布
    #     results['total_signals'] = len(signals_df[signals_df['trading_signal'] != 'NO_SIGNAL'])
    #     results['buy_signals'] = len(signals_df[signals_df['signal_direction'] == 1])
    #     results['sell_signals'] = len(signals_df[signals_df['signal_direction'] == -1])
    #     results['strong_signals'] = len(signals_df[signals_df['signal_strength'] >= 7])
    #
    #     # 计算信号准确率 (简化版)
    #     if 'predicted_future_return_3' in signals_df.columns:
    #         predicted_returns = signals_df['predicted_future_return_3'].fillna(0)
    #         signal_directions = signals_df['signal_direction'].fillna(0)
    #
    #         # 计算方向准确率
    #         correct_predictions = ((predicted_returns > 0) & (signal_directions > 0)) | \
    #                               ((predicted_returns < 0) & (signal_directions < 0))
    #
    #         valid_predictions = signal_directions != 0
    #         if valid_predictions.sum() > 0:
    #             results['win_rate'] = correct_predictions[valid_predictions].mean()
    #             results['avg_return_per_signal'] = abs(predicted_returns[valid_predictions]).mean()
    #
    #     print(f"📈 回测结果:")
    #     print(f"  总信号数: {results['total_signals']}")
    #     print(f"  买入信号: {results['buy_signals']}")
    #     print(f"  卖出信号: {results['sell_signals']}")
    #     print(f"  强信号数: {results['strong_signals']}")
    #     print(f"  胜率: {results['win_rate']:.2%}")
    #     print(f"  平均预期收益: {results['avg_return_per_signal']:.4%}")
    #
    #     return results


class ProfessionalBacktester:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def backtest(self, signals_df, price_df):
        """专业级回测"""
        results = {
            'trades': [],
            'equity_curve': [],
            'positions': []
        }

        capital = self.initial_capital
        position = 0
        entry_price = 0

        for idx, row in signals_df.iterrows():
            current_price = price_df.loc[idx, 'close']

            # 执行交易逻辑
            if row['trading_signal'].startswith('STRONG_BUY') and position == 0:
                # 开多仓
                position_size = row['position_size'] * capital / current_price
                commission_paid = position_size * current_price * self.commission

                position = position_size
                entry_price = current_price
                capital -= commission_paid

                results['trades'].append({
                    'timestamp': idx,
                    'type': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'commission': commission_paid
                })

            elif row['trading_signal'].startswith('STRONG_SELL') and position > 0:
                # 平仓
                exit_value = position * current_price
                commission_paid = exit_value * self.commission
                pnl = exit_value - (position * entry_price) - commission_paid

                capital += exit_value - commission_paid

                results['trades'].append({
                    'timestamp': idx,
                    'type': 'SELL',
                    'price': current_price,
                    'size': position,
                    'pnl': pnl,
                    'return': pnl / (position * entry_price)
                })

                position = 0

            # 记录权益曲线
            current_value = capital + position * current_price
            results['equity_curve'].append({
                'timestamp': idx,
                'capital': capital,
                'position_value': position * current_price,
                'total_value': current_value
            })

        # 计算性能指标
        results['metrics'] = self._calculate_metrics(results)
        return results

    def _calculate_metrics(self, results):
        """计算回测指标"""
        equity_curve = pd.DataFrame(results['equity_curve'])
        trades = pd.DataFrame(results['trades'])

        # 计算收益率
        total_return = (equity_curve['total_value'].iloc[-1] /
                        self.initial_capital - 1)

        # 计算夏普比率
        returns = equity_curve['total_value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # 计算最大回撤
        cummax = equity_curve['total_value'].cummax()
        drawdown = (equity_curve['total_value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 胜率
        winning_trades = trades[trades['pnl'] > 0] if 'pnl' in trades else pd.DataFrame()
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0



        backtest_result = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_return': trades['return'].mean() if 'return' in trades else 0
        }

        logBot.info(backtest_result)
