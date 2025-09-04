#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 16:26     Xsu         1.0         None
'''
from features.base import ExtractFeatureBase
import pandas as pd
import numpy as np
class VolatilityStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Volatility_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        features = pd.DataFrame(index=df_tech.index)
        # 真实波动率
        for period in [5, 10, 20]:
            returns = df_tech['close'].pct_change()
            features[f'realized_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)

        # ATR相关特征
        features['atr_ratio'] = df_tech['atr_7'] / df_tech['atr_14']
        features['atr_percentile'] = df_tech['atr_14'].rolling(50).rank(pct=True)

        # 价格范围特征
        features['daily_range'] = (df_tech['high'] - df_tech['low']) / df_tech['close']
        features['body_ratio'] = abs(df_tech['close'] - df_tech['open']) / (df_tech['high'] - df_tech['low'] + 1e-8)

        # 波动率状态
        features['vol_regime'] = self._identify_volatility_regime(df_tech)

        # Gap特征
        features['gap_up'] = np.where(df_tech['open'] > df_tech['close'].shift(1),
                                      (df_tech['open'] - df_tech['close'].shift(1)) / df_tech['close'].shift(1), 0)
        features['gap_down'] = np.where(df_tech['open'] < df_tech['close'].shift(1),
                                        (df_tech['close'].shift(1) - df_tech['open']) / df_tech['close'].shift(1), 0)

        # 布林带波动率
        features['bb_squeeze'] = np.where(df_tech['bb_width'] < df_tech['bb_width'].rolling(20).quantile(0.2), 1, 0)
        features['bb_expansion'] = np.where(df_tech['bb_width'] > df_tech['bb_width'].rolling(20).quantile(0.8), 1, 0)

        return features

    def _identify_volatility_regime(self, df_tech):
        """识别波动率状态"""
        vol_percentile = df_tech['atr_14'].rolling(50).rank(pct=True)
        return np.where(vol_percentile > 0.8, 2,  # 高波动
                        np.where(vol_percentile < 0.2, 0, 1))  # 低波动, 中等波动