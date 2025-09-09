#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/9 13:07     Xsu         1.0         None
'''

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
峰值感知的机器学习模型
专门针对峰值点特征学习优化
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class PeakAwareMLModel:
    """
    峰值感知的机器学习模型
    - 专注于峰值点的特征学习
    - 处理样本不平衡
    - 多模型集成
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.peak_patterns = {}

    def prepare_peak_focused_data(self, df_tech: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        准备峰值聚焦的训练数据
        """
        print("🎯 准备峰值聚焦数据...")

        # 1. 识别峰值点
        peak_mask = df_tech['is_peak'] == 1
        high_peak_mask = df_tech['peak_type'] == 1
        low_peak_mask = df_tech['peak_type'] == -1

        # 2. 创建峰值特定的目标变量
        targets = {}

        # 峰值后的反转概率（更合理的目标）
        targets['peak_reversal_3'] = self._calculate_reversal_probability(df_tech, 3)
        targets['peak_reversal_5'] = self._calculate_reversal_probability(df_tech, 5)

        # 峰值质量评分（基于历史表现）
        targets['peak_quality'] = self._calculate_peak_quality(df_tech, features_df)

        # 峰值后的趋势持续性
        targets['trend_continuation'] = self._calculate_trend_continuation(df_tech)

        # 3. 构造峰值特定特征
        peak_features = self._engineer_peak_specific_features(df_tech, features_df)

        # 4. 样本平衡策略
        balanced_data = self._balance_peak_samples(peak_features, targets, peak_mask)

        return balanced_data, targets

    def _calculate_reversal_probability(self, df_tech: pd.DataFrame, period: int) -> pd.Series:
        """
        计算峰值后的反转概率
        高点后下跌、低点后上涨的概率
        """
        reversal = pd.Series(0, index=df_tech.index)

        for i in range(len(df_tech) - period):
            if df_tech['peak_type'].iloc[i] == 1:  # 高点
                # 检查后续是否下跌
                future_return = (df_tech['close'].iloc[i + period] - df_tech['close'].iloc[i]) / df_tech['close'].iloc[
                    i]
                reversal.iloc[i] = 1 if future_return < -0.01 else 0  # 下跌超过1%

            elif df_tech['peak_type'].iloc[i] == -1:  # 低点
                # 检查后续是否上涨
                future_return = (df_tech['close'].iloc[i + period] - df_tech['close'].iloc[i]) / df_tech['close'].iloc[
                    i]
                reversal.iloc[i] = 1 if future_return > 0.01 else 0  # 上涨超过1%

        return reversal

    def _calculate_peak_quality(self, df_tech: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        计算峰值质量分数
        基于多个维度评估峰值的重要性
        """
        quality_score = pd.Series(0.0, index=df_tech.index)

        # 1. 价格突破程度
        price_breakout = abs(df_tech['close'] - df_tech['sma_20']) / df_tech['sma_20']

        # 2. 成交量异常
        if 'volume' in df_tech.columns:
            vol_ratio = df_tech['volume'] / df_tech['volume'].rolling(20).mean()
        else:
            vol_ratio = 1.0

        # 3. 技术指标极值
        rsi_extreme = np.where(df_tech['rsi_14'] > 70, (df_tech['rsi_14'] - 70) / 30,
                               np.where(df_tech['rsi_14'] < 30, (30 - df_tech['rsi_14']) / 30, 0))

        # 4. 波动率状态
        atr_percentile = df_tech['atr_14'].rolling(50).rank(pct=True)

        # 综合评分
        quality_score = (
                price_breakout * 0.3 +
                np.clip(vol_ratio, 0, 2) / 2 * 0.3 +
                rsi_extreme * 0.2 +
                atr_percentile * 0.2
        )

        # 只保留峰值点的分数
        quality_score[df_tech['is_peak'] == 0] = 0

        # 分类：高质量(>0.7)、中等(0.4-0.7)、低质量(<0.4)
        return pd.cut(quality_score, bins=[0, 0.4, 0.7, 1.0], labels=[0, 1, 2])

    def _calculate_trend_continuation(self, df_tech: pd.DataFrame) -> pd.Series:
        """
        计算趋势延续性
        峰值后趋势是否继续
        """
        continuation = pd.Series(0, index=df_tech.index)

        for i in range(20, len(df_tech) - 10):
            if df_tech['is_peak'].iloc[i] == 1:
                # 峰值前的趋势
                pre_trend = np.polyfit(range(10), df_tech['close'].iloc[i - 10:i].values, 1)[0]
                # 峰值后的趋势
                post_trend = np.polyfit(range(10), df_tech['close'].iloc[i:i + 10].values, 1)[0]

                # 同向为延续，反向为反转
                continuation.iloc[i] = 1 if pre_trend * post_trend > 0 else -1

        return continuation

    def _engineer_peak_specific_features(self, df_tech: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        构造峰值特定特征
        """
        peak_features = features_df.copy()

        # 1. 峰值间距特征
        peak_features['bars_between_peaks'] = self._calculate_peak_spacing(df_tech)

        # 2. 峰值相对强度
        peak_features['relative_peak_height'] = self._calculate_relative_height(df_tech)

        # 3. 峰值形态特征
        peak_features['peak_sharpness'] = self._calculate_peak_sharpness(df_tech)
        peak_features['peak_symmetry'] = self._calculate_peak_symmetry(df_tech)

        # 4. 多时间框架确认
        for tf in [5, 10, 20]:
            peak_features[f'mtf_alignment_{tf}'] = self._check_multi_timeframe_alignment(df_tech, tf)

        # 5. 峰值聚集度
        peak_features['peak_cluster_density'] = self._calculate_peak_clustering(df_tech)

        # 6. 动量背离特征
        peak_features['momentum_divergence'] = self._calculate_momentum_divergence(df_tech)

        # 7. 支撑阻力特征
        peak_features['sr_distance'] = self._calculate_support_resistance_distance(df_tech)

        return peak_features

    def _calculate_peak_spacing(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算峰值间距"""
        spacing = pd.Series(np.nan, index=df_tech.index)
        peak_indices = df_tech[df_tech['is_peak'] == 1].index

        for i in range(1, len(peak_indices)):
            current_idx = peak_indices[i]
            prev_idx = peak_indices[i - 1]
            spacing.loc[current_idx] = (current_idx - prev_idx).days if hasattr(current_idx - prev_idx, 'days') else i

        return spacing.fillna(method='ffill')

    def _calculate_relative_height(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算相对峰值高度"""
        relative_height = pd.Series(0.0, index=df_tech.index)

        for i in range(10, len(df_tech) - 10):
            if df_tech['is_peak'].iloc[i] == 1:
                window_prices = df_tech['close'].iloc[i - 10:i + 11]

                if df_tech['peak_type'].iloc[i] == 1:  # 高点
                    relative_height.iloc[i] = (df_tech['close'].iloc[i] - window_prices.min()) / window_prices.min()
                else:  # 低点
                    relative_height.iloc[i] = (window_prices.max() - df_tech['close'].iloc[i]) / df_tech['close'].iloc[
                        i]

        return relative_height

    def _calculate_peak_sharpness(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算峰值尖锐度"""
        sharpness = pd.Series(0.0, index=df_tech.index)

        for i in range(2, len(df_tech) - 2):
            if df_tech['is_peak'].iloc[i] == 1:
                # 计算峰值点前后的斜率变化
                left_slope = (df_tech['close'].iloc[i] - df_tech['close'].iloc[i - 2]) / 2
                right_slope = (df_tech['close'].iloc[i + 2] - df_tech['close'].iloc[i]) / 2

                sharpness.iloc[i] = abs(left_slope - right_slope)

        return sharpness

    def _calculate_peak_symmetry(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算峰值对称性"""
        symmetry = pd.Series(0.0, index=df_tech.index)

        for i in range(5, len(df_tech) - 5):
            if df_tech['is_peak'].iloc[i] == 1:
                left_profile = df_tech['close'].iloc[i - 5:i].values
                right_profile = df_tech['close'].iloc[i + 1:i + 6].values

                # 归一化
                left_norm = (left_profile - left_profile.min()) / (left_profile.max() - left_profile.min() + 1e-8)
                right_norm = (right_profile - right_profile.min()) / (right_profile.max() - right_profile.min() + 1e-8)

                # 计算对称性（相似度）
                symmetry.iloc[i] = 1 - np.mean(np.abs(left_norm - right_norm[::-1]))

        return symmetry

    def _check_multi_timeframe_alignment(self, df_tech: pd.DataFrame, timeframe: int) -> pd.Series:
        """检查多时间框架对齐"""
        alignment = pd.Series(0, index=df_tech.index)

        # 计算更大时间框架的移动均线
        ma_tf = df_tech['close'].rolling(timeframe).mean()

        # 检查峰值是否与大时间框架趋势一致
        for i in range(timeframe, len(df_tech)):
            if df_tech['is_peak'].iloc[i] == 1:
                if df_tech['peak_type'].iloc[i] == 1:  # 高点
                    # 高点应该在上升趋势中
                    alignment.iloc[i] = 1 if ma_tf.iloc[i] > ma_tf.iloc[i - timeframe] else -1
                else:  # 低点
                    # 低点应该在下降趋势中
                    alignment.iloc[i] = 1 if ma_tf.iloc[i] < ma_tf.iloc[i - timeframe] else -1

        return alignment

    def _calculate_peak_clustering(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算峰值聚集度"""
        clustering = pd.Series(0.0, index=df_tech.index)
        window = 20

        for i in range(window, len(df_tech)):
            window_data = df_tech.iloc[i - window:i + 1]
            peak_count = window_data['is_peak'].sum()
            clustering.iloc[i] = peak_count / window

        return clustering

    def _calculate_momentum_divergence(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算动量背离"""
        divergence = pd.Series(0, index=df_tech.index)

        for i in range(14, len(df_tech)):
            if df_tech['is_peak'].iloc[i] == 1:
                # 价格趋势
                price_trend = np.polyfit(range(14), df_tech['close'].iloc[i - 13:i + 1].values, 1)[0]

                # RSI趋势
                rsi_trend = np.polyfit(range(14), df_tech['rsi_14'].iloc[i - 13:i + 1].values, 1)[0]

                # 背离检测
                if df_tech['peak_type'].iloc[i] == 1:  # 高点
                    # 价格创新高但RSI没有 - 看跌背离
                    divergence.iloc[i] = -1 if price_trend > 0 and rsi_trend < 0 else 0
                else:  # 低点
                    # 价格创新低但RSI没有 - 看涨背离
                    divergence.iloc[i] = 1 if price_trend < 0 and rsi_trend > 0 else 0

        return divergence

    def _calculate_support_resistance_distance(self, df_tech: pd.DataFrame) -> pd.Series:
        """计算到支撑阻力的距离"""
        distance = pd.Series(0.0, index=df_tech.index)

        for i in range(50, len(df_tech)):
            current_price = df_tech['close'].iloc[i]

            # 找出过去50根K线的峰值作为潜在支撑阻力
            past_peaks = df_tech.iloc[i - 50:i][df_tech['is_peak'].iloc[i - 50:i] == 1]['close']

            if len(past_peaks) > 0:
                # 找最近的支撑和阻力
                resistance = past_peaks[past_peaks > current_price]
                support = past_peaks[past_peaks < current_price]

                if len(resistance) > 0:
                    distance.iloc[i] = (resistance.min() - current_price) / current_price
                elif len(support) > 0:
                    distance.iloc[i] = (current_price - support.max()) / current_price

        return distance

    def _balance_peak_samples(self, features: pd.DataFrame, targets: Dict,
                              peak_mask: pd.Series) -> pd.DataFrame:
        """
        平衡峰值和非峰值样本
        """
        print("⚖️ 平衡样本分布...")

        # 分离峰值和非峰值样本
        peak_samples = features[peak_mask]
        non_peak_samples = features[~peak_mask]

        # 策略1：对非峰值样本进行欠采样
        n_peak = len(peak_samples)
        n_select = min(n_peak * 5, len(non_peak_samples))  # 最多5:1的比例

        # 智能采样：选择接近峰值的样本
        selected_non_peak = self._smart_undersample(non_peak_samples, n_select)

        # 合并数据
        balanced_features = pd.concat([peak_samples, selected_non_peak])

        print(f"  峰值样本: {len(peak_samples)}")
        print(f"  非峰值样本: {len(selected_non_peak)}")
        print(f"  总样本: {len(balanced_features)}")

        return balanced_features

    def _smart_undersample(self, non_peak_samples: pd.DataFrame, n_select: int) -> pd.DataFrame:
        """
        智能欠采样：优先选择接近峰值的样本
        """
        if len(non_peak_samples) <= n_select:
            return non_peak_samples

        # 计算每个样本的"峰值相似度"分数
        scores = pd.Series(0.0, index=non_peak_samples.index)

        # 基于多个指标计算相似度
        if 'rsi_14' in non_peak_samples.columns:
            rsi_extreme = np.abs(non_peak_samples['rsi_14'] - 50) / 50
            scores += rsi_extreme * 0.3

        if 'bb_position' in non_peak_samples.columns:
            bb_extreme = np.abs(non_peak_samples['bb_position'] - 0.5) * 2
            scores += bb_extreme * 0.3

        if 'atr_percentile' in non_peak_samples.columns:
            scores += non_peak_samples['atr_percentile'] * 0.2

        if 'volume_spike' in non_peak_samples.columns:
            scores += non_peak_samples['volume_spike'] * 0.2

        # 选择分数最高的样本
        top_indices = scores.nlargest(n_select).index

        return non_peak_samples.loc[top_indices]

    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series,
                             feature_cols: List[str]) -> Dict:
        """
        训练集成模型
        """
        print("🤖 训练峰值感知集成模型...")

        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[feature_cols].fillna(0))

        # 1. XGBoost模型
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1,
            random_state=42
        )

        # 2. LightGBM模型
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42
        )

        # 3. 随机森林（专注于峰值特征）
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )

        # 4. 梯度提升
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.7,
            random_state=42
        )

        # 5. 集成投票分类器
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',
            weights=[2, 2, 1, 1]  # XGBoost和LightGBM权重更高
        )

        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=5)

        # 训练和评估
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=tscv, scoring='roc_auc')
        print(f"  交叉验证AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 训练最终模型
        ensemble.fit(X_scaled, y)

        # 提取特征重要性（从随机森林）
        rf_model.fit(X_scaled, y)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n📊 峰值特征重要性 Top 10:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return {
            'ensemble': ensemble,
            'scaler': scaler,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores
        }

    def detect_peak_patterns(self, X: pd.DataFrame, model: Dict) -> pd.DataFrame:
        """
        检测峰值模式
        """
        ensemble = model['ensemble']
        scaler = model['scaler']

        # 预测
        X_scaled = scaler.transform(X.fillna(0))

        # 获取概率预测
        probabilities = ensemble.predict_proba(X_scaled)

        # 生成信号
        signals = pd.DataFrame(index=X.index)

        # 峰值概率
        signals['peak_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

        # 峰值强度分级
        signals['peak_signal'] = pd.cut(
            signals['peak_probability'],
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            labels=['NO_PEAK', 'WEAK_PEAK', 'MODERATE_PEAK', 'STRONG_PEAK', 'EXTREME_PEAK']
        )

        # 使用Isolation Forest检测异常峰值
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42
        )
        anomaly_scores = iso_forest.fit_predict(X_scaled)
        signals['is_anomaly_peak'] = anomaly_scores == -1

        return signals