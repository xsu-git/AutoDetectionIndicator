#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/4 16:37     Xsu         1.0         峰值时间特征 (10维)
'''
from features.base import ExtractFeatureBase
import pandas as pd
import numpy as np
class TemporalStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Temporal_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        features = pd.DataFrame(index=df_tech.index)
        # 时间周期特征
        features['hour'] = df_tech.index.hour
        features['day_of_week'] = df_tech.index.dayofweek
        features['is_weekend'] = (df_tech.index.dayofweek >= 5).astype(int)

        # 市场开盘时间特征（加密货币24小时，但仍有高低活跃期）
        features['is_asian_session'] = ((df_tech.index.hour >= 0) & (df_tech.index.hour < 8)).astype(int)
        features['is_european_session'] = ((df_tech.index.hour >= 8) & (df_tech.index.hour < 16)).astype(int)
        features['is_us_session'] = ((df_tech.index.hour >= 16) & (df_tech.index.hour < 24)).astype(int)

        # 周期性模式
        features['hour_sin'] = np.sin(2 * np.pi * df_tech.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df_tech.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * df_tech.index.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * df_tech.index.dayofweek / 7)

        return features