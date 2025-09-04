#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 16:35     Xsu         1.0         市场微结构特征 (15维)
'''
from features.base import ExtractFeatureBase
import pandas as pd
import numpy as np
class MicroStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Micro_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        features = pd.DataFrame(index=df_tech.index)
        # 买卖压力
        features['buying_pressure'] = (df_tech['close'] - df_tech['low']) / (df_tech['high'] - df_tech['low'] + 1e-8)
        features['selling_pressure'] = (df_tech['high'] - df_tech['close']) / (df_tech['high'] - df_tech['low'] + 1e-8)

        # K线形态
        features['doji'] = (
                abs(df_tech['close'] - df_tech['open']) / (df_tech['high'] - df_tech['low'] + 1e-8) < 0.1).astype(
            int)
        features['hammer'] = self._identify_hammer_pattern(df_tech)
        features['engulfing'] = self._identify_engulfing_pattern(df_tech)

        # 价格效率
        features['price_efficiency_5'] = self._calculate_price_efficiency(df_tech['close'], 5)
        features['price_efficiency_10'] = self._calculate_price_efficiency(df_tech['close'], 10)

        # 流动性指标
        features['bid_ask_proxy'] = (df_tech['high'] - df_tech['low']) / df_tech['volume']

        # 价格冲击
        features['price_impact'] = abs(df_tech['close'].pct_change()) / (df_tech['volume'].pct_change() + 1e-8)

        return features

    def _identify_hammer_pattern(self, df_tech):
        """识别锤子线形态"""
        body_size = abs(df_tech['close'] - df_tech['open'])
        lower_shadow = df_tech['open'].combine(df_tech['close'], min) - df_tech['low']
        upper_shadow = df_tech['high'] - df_tech['open'].combine(df_tech['close'], max)

        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)

    def _identify_engulfing_pattern(self, df_tech):
        """识别吞没形态"""
        prev_body = abs(df_tech['close'].shift(1) - df_tech['open'].shift(1))
        curr_body = abs(df_tech['close'] - df_tech['open'])

        bullish_engulf = ((df_tech['close'] > df_tech['open']) &
                          (df_tech['close'].shift(1) < df_tech['open'].shift(1)) &
                          (curr_body > prev_body)).astype(int)

        bearish_engulf = ((df_tech['close'] < df_tech['open']) &
                          (df_tech['close'].shift(1) > df_tech['open'].shift(1)) &
                          (curr_body > prev_body)).astype(int)

        return bullish_engulf - bearish_engulf

    def _calculate_price_efficiency(self, price_series, period):
        """计算价格效率"""
        price_change = abs(price_series.diff(period))
        path_length = abs(price_series.diff()).rolling(period).sum()
        return price_change / (path_length + 1e-8)

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