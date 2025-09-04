#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 15:54     Xsu         1.0         峰值结构特征 (20维)
'''
from features.base import ExtractFeatureBase
import pandas as pd
import numpy as np
class PeakStructureFeature(ExtractFeatureBase):

    def __init__(self, ma_type: str = 'Peak_Structure'):
        super().__init__(f"{ma_type}")
        self.ma_type = ma_type.upper()

    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:

        features = pd.DataFrame(index=df_tech.index)

        # 峰值密度特征
        for window in [10, 20, 50]:
            peak_density = df_tech['is_peak'].rolling(window=window).sum()
            features[f'peak_density_{window}'] = peak_density / window

        # 峰值强度分布
        for window in [10, 20]:
            features[f'peak_score_mean_{window}'] = df_tech['peak_score'].rolling(window=window).mean()
            features[f'peak_score_std_{window}'] = df_tech['peak_score'].rolling(window=window).std()

        # 高低点比率
        for window in [20, 50]:
            high_count = (df_tech['peak_type'] == 1).rolling(window=window).sum()
            low_count = (df_tech['peak_type'] == -1).rolling(window=window).sum()
            features[f'high_low_ratio_{window}'] = high_count / (low_count + 1)

        # 距离上次峰值的时间
        features['bars_since_last_peak'] = self._bars_since_last_event(df_tech['is_peak'])
        features['bars_since_last_high'] = self._bars_since_last_event(df_tech['peak_type'] == 1)
        features['bars_since_last_low'] = self._bars_since_last_event(df_tech['peak_type'] == -1)

        # 峰值价格距离
        features['price_to_last_high'] = self._price_distance_to_last_peak(df_tech, 1)
        features['price_to_last_low'] = self._price_distance_to_last_peak(df_tech, -1)

        # 峰值趋势特征
        features['peak_trend_5'] = self._calculate_peak_trend(df_tech, 5)
        features['peak_trend_10'] = self._calculate_peak_trend(df_tech, 10)

        return features

    def _bars_since_last_event(self, condition_series: pd.Series) -> pd.Series:
        """
        计算距离上次事件的K线数（向量化）。
        - 事件：condition_series 为 True 的位置
        - 事件之前：NaN
        - 事件点：0
        - 事件之后：递增 1,2,3...
        """
        n = len(condition_series)
        idx = condition_series.index
        # 将 NaN 当作 False（若你需要保持 NaN 特殊含义，可自行调整）
        ev = condition_series.fillna(False).to_numpy(dtype=bool)

        ar = np.arange(n, dtype=np.int64)
        # 把事件位置设为自身索引，否则为 -1；再做前缀最大值，得到“最近一次事件的索引”
        last_idx = np.where(ev, ar, -1)
        last_idx = np.maximum.accumulate(last_idx)

        out = np.full(n, np.nan, dtype=float)
        m = last_idx != -1
        out[m] = ar[m] - last_idx[m]
        return pd.Series(out, index=idx, dtype=float)

    def _price_distance_to_last_peak(self, df_tech, peak_type):
        """
        纯NumPy实现的价格距离计算（最高性能版本）
        """
        n = len(df_tech)
        idx = df_tech.index

        # 转换为NumPy数组
        is_peak = (df_tech['peak_type'] == peak_type).to_numpy()
        prices = df_tech['close'].to_numpy(dtype=float)

        # 手动实现forward fill，避免创建Series的开销
        last_peak_prices = np.full(n, np.nan, dtype=float)
        current_peak = np.nan

        for i in range(n):
            if is_peak[i]:
                current_peak = prices[i]
            last_peak_prices[i] = current_peak

        # 向量化计算
        result = np.full(n, np.nan, dtype=float)
        result[is_peak] = 0.0

        # 计算非峰值位置的距离
        valid_mask = ~np.isnan(last_peak_prices) & ~is_peak
        result[valid_mask] = (prices[valid_mask] - last_peak_prices[valid_mask]) / last_peak_prices[valid_mask]

        return pd.Series(result, index=idx, dtype=float)

    def _calculate_peak_trend(self, df_tech, window):
        """
        纯向量化版本（适合峰值密集的场景）
        通过预处理峰值序列，减少NaN处理开销
        """
        idx = df_tech.index

        # 1. 提取所有峰值及其位置
        peak_mask = df_tech['is_peak'] == 1
        peak_positions = np.where(peak_mask)[0]
        peak_values = df_tech.loc[peak_mask, 'close'].values

        if len(peak_values) < 2:
            return pd.Series(np.nan, index=idx, dtype=float)

        # 2. 为每个峰值计算趋势（向前看window个峰值）
        result = np.full(len(df_tech), np.nan, dtype=float)

        for i in range(len(peak_positions)):
            current_pos = peak_positions[i]

            # 获取当前及之前的峰值（最多window个）
            end_idx = i + 1
            start_idx = max(0, end_idx - window)

            if end_idx - start_idx >= 2:  # 至少需要2个峰值
                # 提取相关峰值
                relevant_positions = peak_positions[start_idx:end_idx]
                relevant_values = peak_values[start_idx:end_idx]

                # 计算斜率
                x = np.arange(len(relevant_values))
                slope = np.polyfit(x, relevant_values, 1)[0]

                # 将结果赋给当前峰值位置
                result[current_pos] = slope

        # 3. 向前填充到所有位置（可选，根据需求决定）
        result_series = pd.Series(result, index=idx)
        # result_series = result_series.fillna(method='ffill')  # 如果需要填充

        return result_series


