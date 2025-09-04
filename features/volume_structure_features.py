#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 16:28     Xsu         1.0         None
'''
from features.base import ExtractFeatureBase
import pandas as pd
import talib
class VolumeStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Volume_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        features = pd.DataFrame(index=df_tech.index)
        # 成交量移动均线
        for period in [5, 10, 20]:
            features[f'vol_sma_{period}'] = df_tech['volume'].rolling(period).mean()
            features[f'vol_ratio_{period}'] = df_tech['volume'] / features[f'vol_sma_{period}']

        # 成交量价格趋势
        # features['vpt'] = talib.VPT(df_tech['close'], df_tech['volume'])
        features['vpt'] = self._calculate_vpt_optimized(df_tech['close'], df_tech['volume'])
        features['obv'] = talib.OBV(df_tech['close'], df_tech['volume'])

        # 资金流指标
        features['mfi'] = talib.MFI(df_tech['high'], df_tech['low'], df_tech['close'], df_tech['volume'])
        features['ad'] = talib.AD(df_tech['high'], df_tech['low'], df_tech['close'], df_tech['volume'])

        # 成交量分布
        features['vol_percentile'] = df_tech['volume'].rolling(50).rank(pct=True)

        # 异常成交量
        vol_threshold = df_tech['volume'].rolling(20).mean() + 2 * df_tech['volume'].rolling(20).std()
        features['volume_spike'] = (df_tech['volume'] > vol_threshold).astype(int)

        # 价量关系
        price_change = df_tech['close'].pct_change()
        features['price_volume_corr_10'] = price_change.rolling(10).corr(df_tech['volume'].pct_change())

        # VWAP相关
        if 'vwap' not in df_tech.columns:
            df_tech['vwap'] = (df_tech['close'] * df_tech['volume']).rolling(20).sum() / df_tech['volume'].rolling(
                20).sum()

        features['price_to_vwap'] = df_tech['close'] / df_tech['vwap'] - 1

        return features

    def _calculate_vpt_optimized(self, close, volume):
        """
        VPT计算的优化版本 - 使用pandas内置函数
        """
        # 计算价格变化率
        price_pct_change = close.pct_change()

        # 计算VPT增量
        vpt_increments = volume * price_pct_change

        # 累积求和，第一个值为0
        vpt = vpt_increments.fillna(0).cumsum()

        return vpt