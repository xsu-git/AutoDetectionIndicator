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

        # 3) 回归预测
        for target in ['future_return_3', 'future_return_5']:
            if target in self.models:
                pred = pd.Series(self.models[target].predict(X_scaled), index=idx)
                signals_df.loc[idx, f'predicted_{target}'] = pred

        # 4) 分类预测 + 置信度
        for target in ['direction_3', 'direction_5', 'significant_move_3', 'significant_move_5']:
            if target in self.models:
                pred = pd.Series(self.models[target].predict(X_scaled), index=idx)
                signals_df.loc[idx, f'predicted_{target}'] = pred

                # 一些模型（如 SVC 概率关闭）可能没有 predict_proba，做个保护
                if hasattr(self.models[target], "predict_proba"):
                    proba = self.models[target].predict_proba(X_scaled)
                    max_prob = pd.Series(np.max(proba, axis=1), index=idx)
                    signals_df.loc[idx, f'{target}_confidence'] = max_prob

        # 5) 组合信号
        signals_df = self._generate_composite_signals(signals_df)
        return signals_df

        # 回归预测 - 预期收益率
        # for target in ['future_return_3', 'future_return_5']:
        #     if target in self.models:
        #         predictions = self.models[target].predict(X_scaled)
        #         signals_df[f'predicted_{target}'] = predictions
        #
        # # 分类预测 - 方向和概率
        # for target in ['direction_3', 'direction_5', 'significant_move_3', 'significant_move_5']:
        #     if target in self.models:
        #
        #         predictions = self.models[target].predict(X_scaled)
        #         probabilities = self.models[target].predict_proba(X_scaled)
        #
        #         signals_df[f'predicted_{target}'] = predictions
        #
        #         # 获取最高概率
        #         max_prob = np.max(probabilities, axis=1)
        #         signals_df[f'{target}_confidence'] = max_prob
        #
        # # 生成综合交易信号
        # signals_df = self._generate_composite_signals(signals_df)
        #
        # return signals_df

    def _generate_composite_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """生成综合交易信号"""

        # 1. 强度评分系统
        signals_df['bull_strength'] = 0.0
        signals_df['bear_strength'] = 0.0

        # 基于收益率预测的强度
        if 'predicted_future_return_3' in signals_df.columns:
            signals_df['bull_strength'] += np.clip(signals_df['predicted_future_return_3'] * 100, 0, 5)
            signals_df['bear_strength'] += np.clip(-signals_df['predicted_future_return_3'] * 100, 0, 5)

        if 'predicted_future_return_5' in signals_df.columns:
            signals_df['bull_strength'] += np.clip(signals_df['predicted_future_return_5'] * 100, 0, 3)
            signals_df['bear_strength'] += np.clip(-signals_df['predicted_future_return_5'] * 100, 0, 3)

        # 基于方向预测的强度
        if 'predicted_direction_3' in signals_df.columns and 'direction_3_confidence' in signals_df.columns:
            direction_strength = signals_df['direction_3_confidence'] * 3
            signals_df['bull_strength'] += np.where(signals_df['predicted_direction_3'] == 1, direction_strength, 0)
            signals_df['bear_strength'] += np.where(signals_df['predicted_direction_3'] == -1, direction_strength, 0)

        # 基于显著性预测的强度
        if 'significant_move_3_confidence' in signals_df.columns:
            signals_df['bull_strength'] += signals_df['significant_move_3_confidence'] * 2
            signals_df['bear_strength'] += signals_df['significant_move_3_confidence'] * 2

        # 2. 信号等级分类
        total_strength = signals_df['bull_strength'] + signals_df['bear_strength']
        net_strength = signals_df['bull_strength'] - signals_df['bear_strength']

        # 信号强度等级 (0-10分)
        signals_df['signal_strength'] = np.clip(total_strength, 0, 10)

        # 信号方向 (-1, 0, 1)
        signals_df['signal_direction'] = np.where(
            net_strength > 1, 1,  # 看涨
            np.where(net_strength < -1, -1, 0)  # 看跌, 中性
        )

        # 3. 交易信号等级
        def categorize_signal(row):
            strength = row['signal_strength']
            direction = row['signal_direction']

            if strength >= 7:
                return f"STRONG_{'BUY' if direction > 0 else 'SELL' if direction < 0 else 'HOLD'}"
            elif strength >= 5:
                return f"MODERATE_{'BUY' if direction > 0 else 'SELL' if direction < 0 else 'HOLD'}"
            elif strength >= 3:
                return f"WEAK_{'BUY' if direction > 0 else 'SELL' if direction < 0 else 'HOLD'}"
            else:
                return "NO_SIGNAL"

        signals_df['trading_signal'] = signals_df.apply(categorize_signal, axis=1)

        # 4. 风险调整后的仓位建议
        signals_df['position_size'] = self._calculate_position_size(signals_df)

        # 5. 止损止盈建议
        signals_df = self._calculate_stop_take_levels(signals_df)

        return signals_df

    def _calculate_position_size(self, signals_df: pd.DataFrame) -> pd.Series:
        """计算建议仓位大小 (0-1之间)"""
        base_size = signals_df['signal_strength'] / 10 * 0.3  # 基础仓位最大30%

        # 信号置信度调整
        confidence_avg = (
                                 signals_df.get('direction_3_confidence', 0.5) +
                                 signals_df.get('significant_move_3_confidence', 0.5)
                         ) / 2

        adjusted_size = base_size * confidence_avg

        return np.clip(adjusted_size, 0, 0.5)  # 最大50%仓位

    def _calculate_stop_take_levels(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """计算止损止盈水平"""

        # 基于预期收益率的止盈
        expected_return_3 = signals_df.get('predicted_future_return_3', 0)
        expected_return_5 = signals_df.get('predicted_future_return_5', 0)

        # 动态止盈 (预期收益的80%)
        signals_df['take_profit_pct'] = np.clip(
            np.maximum(abs(expected_return_3), abs(expected_return_5)) * 0.8,
            0.005,  # 最小0.5%
            0.05  # 最大5%
        )

        # 动态止损 (预期收益的40%，但最大2%)
        signals_df['stop_loss_pct'] = np.clip(
            signals_df['take_profit_pct'] * 0.4,
            0.003,  # 最小0.3%
            0.02  # 最大2%
        )

        # 计算具体价位
        current_price = signals_df['close_price']

        signals_df['take_profit_long'] = current_price * (1 + signals_df['take_profit_pct'])
        signals_df['stop_loss_long'] = current_price * (1 - signals_df['stop_loss_pct'])

        signals_df['take_profit_short'] = current_price * (1 - signals_df['take_profit_pct'])
        signals_df['stop_loss_short'] = current_price * (1 + signals_df['stop_loss_pct'])

        return signals_df

    def analyze_feature_importance(self) -> dict:
        """分析特征重要性"""
        if not self.is_trained:
            return {}

        importance_dict = {}

        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # 获取特征重要性
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                importance_dict[model_name] = feature_importance.head(20)

                print(f"\n🔍 {model_name} - Top 10 重要特征:")
                for idx, row in feature_importance.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_dict

    def backtest_signals(self, signals_df: pd.DataFrame) -> dict:
        """回测交易信号"""
        print("📊 执行信号回测...")

        # 简单回测逻辑
        results = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'strong_signals': 0,
            'win_rate': 0.0,
            'avg_return_per_signal': 0.0
        }

        # 统计信号分布
        results['total_signals'] = len(signals_df[signals_df['trading_signal'] != 'NO_SIGNAL'])
        results['buy_signals'] = len(signals_df[signals_df['signal_direction'] == 1])
        results['sell_signals'] = len(signals_df[signals_df['signal_direction'] == -1])
        results['strong_signals'] = len(signals_df[signals_df['signal_strength'] >= 7])

        # 计算信号准确率 (简化版)
        if 'predicted_future_return_3' in signals_df.columns:
            predicted_returns = signals_df['predicted_future_return_3'].fillna(0)
            signal_directions = signals_df['signal_direction'].fillna(0)

            # 计算方向准确率
            correct_predictions = ((predicted_returns > 0) & (signal_directions > 0)) | \
                                  ((predicted_returns < 0) & (signal_directions < 0))

            valid_predictions = signal_directions != 0
            if valid_predictions.sum() > 0:
                results['win_rate'] = correct_predictions[valid_predictions].mean()
                results['avg_return_per_signal'] = abs(predicted_returns[valid_predictions]).mean()

        print(f"📈 回测结果:")
        print(f"  总信号数: {results['total_signals']}")
        print(f"  买入信号: {results['buy_signals']}")
        print(f"  卖出信号: {results['sell_signals']}")
        print(f"  强信号数: {results['strong_signals']}")
        print(f"  胜率: {results['win_rate']:.2%}")
        print(f"  平均预期收益: {results['avg_return_per_signal']:.4%}")

        return results


