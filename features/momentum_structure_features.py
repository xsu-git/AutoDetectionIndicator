#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 15:25     Xsu         1.0         价格动量特征 (25维)
'''
from features.base import ExtractFeatureBase
import pandas as pd
import numpy as np
import talib
class MomentumStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Momentum_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        features = pd.DataFrame(index=df_tech.index)

        # 多周期价格变化率
        for period in [1, 3, 5, 10, 20]:
            features[f'price_change_{period}'] = df_tech['close'].pct_change(period)
            features[f'high_change_{period}'] = df_tech['high'].pct_change(period)
            features[f'low_change_{period}'] = df_tech['low'].pct_change(period)

        # 价格动量指标
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = talib.MOM(df_tech['close'], timeperiod=period)
            features[f'roc_{period}'] = talib.ROC(df_tech['close'], timeperiod=period)

        # 相对强弱指标
        features['rsi_momentum'] = df_tech['rsi_14'].diff()
        features['rsi_acceleration'] = features['rsi_momentum'].diff()

        # 价格加速度
        features['price_acceleration'] = df_tech['close'].diff().diff()

        # 动量发散
        features['momentum_divergence_5'] = self._calculate_momentum_divergence(df_tech, 5)
        features['momentum_divergence_10'] = self._calculate_momentum_divergence(df_tech, 10)

        return features

    def _calculate_momentum_divergence(self, df_tech, period):
        """
        使用Pandas优化的版本（中等性能，代码简洁）
        """

        # 使用自定义的快速线性回归函数
        def fast_linregress(x):
            """快速线性回归，只返回斜率"""
            if len(x) < 2:
                return np.nan
            n = len(x)
            x_coords = np.arange(n)
            sum_x = np.sum(x_coords)
            sum_y = np.sum(x)
            sum_xy = np.sum(x_coords * x)
            sum_x2 = np.sum(x_coords * x_coords)

            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-12:
                return np.nan

            return (n * sum_xy - sum_x * sum_y) / denominator

        # 计算趋势
        price_trend = df_tech['close'].rolling(period).apply(fast_linregress, raw=True)
        rsi_trend = df_tech['rsi_14'].rolling(period).apply(fast_linregress, raw=True)

        # 背离信号：价格和RSI趋势方向相反
        return np.where((price_trend > 0) & (rsi_trend < 0), 1,  # 看跌背离
                        np.where((price_trend < 0) & (rsi_trend > 0), -1, 0))  # 看涨背离