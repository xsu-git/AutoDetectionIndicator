#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 16:31     Xsu         1.0         技术指标特征 (25维)
'''
from features.base import ExtractFeatureBase
import pandas as pd
import talib
import numpy as np
class TechnicalStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Technical_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        features = pd.DataFrame(index=df_tech.index)
        # RSI特征族
        features['rsi_oversold'] = (df_tech['rsi_14'] < 30).astype(int)
        features['rsi_overbought'] = (df_tech['rsi_14'] > 70).astype(int)
        features['rsi_divergence'] = self._calculate_rsi_divergence(df_tech)

        # MACD特征
        features['macd_signal_cross'] = self._detect_signal_cross(df_tech['macd'], df_tech['macd_signal'])
        features['macd_histogram_peak'] = self._detect_histogram_peaks(df_tech['macd_hist'])
        features['macd_zero_cross'] = self._detect_signal_cross(df_tech['macd'], pd.Series(0, index=df_tech.index))

        # 移动均线特征
        features['ma_cross_5_20'] = self._detect_signal_cross(df_tech['sma_5'], df_tech['sma_20'])
        features['ma_cross_10_50'] = self._detect_signal_cross(df_tech['sma_10'], df_tech['sma_50'])

        # 布林带特征
        features['bb_upper_touch'] = (df_tech['high'] >= df_tech['bb_upper']).astype(int)
        features['bb_lower_touch'] = (df_tech['low'] <= df_tech['bb_lower']).astype(int)
        features['bb_middle_cross'] = self._detect_signal_cross(df_tech['close'], df_tech['bb_middle'])

        # 支撑阻力突破
        features['support_break'] = self._detect_support_resistance_break(df_tech, 'support')
        features['resistance_break'] = self._detect_support_resistance_break(df_tech, 'resistance')

        # 趋势强度
        features['trend_strength'] = self._calculate_trend_strength(df_tech)

        # 市场状态
        features['market_regime'] = self._identify_market_regime(df_tech)

        return features

    def _calculate_rsi_divergence(self, df_tech):
        """计算RSI背离"""
        return self._calculate_momentum_divergence(df_tech, 14)

    def _detect_signal_cross(self, series1, series2):
        """检测信号交叉"""
        cross_up = ((series1 > series2) & (series1.shift(1) <= series2.shift(1))).astype(int)
        cross_down = ((series1 < series2) & (series1.shift(1) >= series2.shift(1))).astype(int)
        return cross_up - cross_down

    def _detect_histogram_peaks(self, hist_series):
        """检测MACD柱状图峰值"""
        peaks = (hist_series > hist_series.shift(1)) & (hist_series > hist_series.shift(-1))
        troughs = (hist_series < hist_series.shift(1)) & (hist_series < hist_series.shift(-1))
        return peaks.astype(int) - troughs.astype(int)

    def _detect_support_resistance_break(self, df_tech, sr_type):
        """检测支撑阻力突破"""
        if sr_type == 'resistance':
            resistance = df_tech['high'].rolling(20).max()
            return (df_tech['close'] > resistance.shift(1)).astype(int)
        else:
            support = df_tech['low'].rolling(20).min()
            return (df_tech['close'] < support.shift(1)).astype(int)

    def _calculate_trend_strength(self, df_tech):
        """计算趋势强度"""
        adx = talib.ADX(df_tech['high'], df_tech['low'], df_tech['close'], timeperiod=14)
        return adx / 100  # 标准化到0-1

    def _identify_market_regime(self, df_tech):
        """识别市场状态"""
        sma_20 = df_tech['sma_20']
        sma_50 = df_tech['sma_50'] if 'sma_50' in df_tech.columns else df_tech['sma_20']

        return np.where(sma_20 > sma_50, 1,  # 牛市
                        np.where(sma_20 < sma_50, -1, 0))  # 熊市, 震荡

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