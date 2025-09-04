#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 10:25     Xsu         1.0         None
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
import talib
from scipy import stats
from scipy.signal import savgol_filter
from process.loader import load_process_data
from utils.data_preprocess_util import restore_hook,build_data_dir
from pathlib import Path
import json
from utils import logBot
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


class PeakFeatureEngineer:
    """
    峰值特征工程师 - 华尔街量化级别
    从检测到的峰值点位中提取130+维交易特征
    """

    def __init__(self, lookback_periods=[3, 5, 8, 13, 21]):
        self.lookback_periods = lookback_periods
        self.feature_names = []

    def extract_comprehensive_features(self, df: pd.DataFrame, peaks_results: dict) -> pd.DataFrame:
        """
        提取综合特征矩阵

        特征类别：
        1. 峰值结构特征 (20维)
        2. 价格动量特征 (25维) 
        3. 波动率结构特征 (15维)
        4. 成交量特征 (20维)
        5. 技术指标特征 (25维)
        6. 市场微结构特征 (15维)
        7. 峰值时间特征 (10维)
        """
        logBot.info("Start Extraction feature engineering...")
        df_tech = self._calculate_technical_base(df)
        logBot.info("Compute Base Technical indicators Finish")

        # 创建峰值标记
        df_tech['is_peak'] = 0
        df_tech['peak_type'] = 0  # -1: low, 0: none, 1: high
        df_tech['peak_score'] = 0.0

        # 标记高低点
        for high in peaks_results['highs']:
            idx = high['index']
            if idx < len(df_tech):
                df_tech.iloc[idx, df_tech.columns.get_loc('is_peak')] = 1
                df_tech.iloc[idx, df_tech.columns.get_loc('peak_type')] = 1
                df_tech.iloc[idx, df_tech.columns.get_loc('peak_score')] = high['score']

        for low in peaks_results['lows']:
            idx = low['index']
            if idx < len(df_tech):
                df_tech.iloc[idx, df_tech.columns.get_loc('is_peak')] = 1
                df_tech.iloc[idx, df_tech.columns.get_loc('peak_type')] = -1
                df_tech.iloc[idx, df_tech.columns.get_loc('peak_score')] = low['score']

        # 提取各类特征
        features_df = pd.DataFrame(index=df_tech.index)

        # 1. 峰值结构特征
        logBot.info("Start extract peak structure features")
        peak_structure_features = self._extract_peak_structure_features(df_tech, peaks_results)



        features_df = pd.concat([features_df, peak_structure_features], axis=1)

        # 2. 价格动量特征
        logBot.info("Start extract Price Momentum features")
        momentum_features = self._extract_momentum_features(df_tech)
        features_df = pd.concat([features_df, momentum_features], axis=1)

        # 3. 波动率结构特征
        logBot.info("Start extract volatility structure features")
        volatility_features = self._extract_volatility_features(df_tech)
        features_df = pd.concat([features_df, volatility_features], axis=1)

        # 4. 成交量特征
        logBot.info("Start extract volume structure features")
        volume_features = self._extract_volume_features(df_tech)
        features_df = pd.concat([features_df, volume_features], axis=1)

        # 5. 技术指标特征
        logBot.info("Start extract technical structure features")
        technical_features = self._extract_technical_features(df_tech)
        features_df = pd.concat([features_df, technical_features], axis=1)

        # 6. 市场微结构特征
        microstructure_features = self._extract_microstructure_features(df_tech)
        features_df = pd.concat([features_df, microstructure_features], axis=1)

        # 7. 峰值时间特征
        temporal_features = self._extract_temporal_features(df_tech)
        features_df = pd.concat([features_df, temporal_features], axis=1)

        # 添加目标变量
        features_df = self._create_target_variables(features_df, df_tech)

        return features_df.dropna()

    def _calculate_technical_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        df_tech = df.copy()

        # 基础价格指标
        df_tech['hl2'] = (df_tech['high'] + df_tech['low']) / 2
        df_tech['hlc3'] = (df_tech['high'] + df_tech['low'] + df_tech['close']) / 3
        df_tech['ohlc4'] = (df_tech['open'] + df_tech['high'] + df_tech['low'] + df_tech['close']) / 4

        # 移动均线族
        for period in [5, 10, 20, 50]:
            df_tech[f'sma_{period}'] = talib.SMA(df_tech['close'], timeperiod=period)
            df_tech[f'ema_{period}'] = talib.EMA(df_tech['close'], timeperiod=period)

        # 波动率指标
        df_tech['atr_14'] = talib.ATR(df_tech['high'], df_tech['low'], df_tech['close'], timeperiod=14)
        df_tech['atr_7'] = talib.ATR(df_tech['high'], df_tech['low'], df_tech['close'], timeperiod=7)

        # 动量指标
        df_tech['rsi_14'] = talib.RSI(df_tech['close'], timeperiod=14)
        df_tech['rsi_7'] = talib.RSI(df_tech['close'], timeperiod=7)

        # MACD
        df_tech['macd'], df_tech['macd_signal'], df_tech['macd_hist'] = talib.MACD(df_tech['close'])

        # 布林带
        df_tech['bb_upper'], df_tech['bb_middle'], df_tech['bb_lower'] = talib.BBANDS(df_tech['close'])
        df_tech['bb_width'] = (df_tech['bb_upper'] - df_tech['bb_lower']) / df_tech['bb_middle']
        df_tech['bb_position'] = (df_tech['close'] - df_tech['bb_lower']) / (df_tech['bb_upper'] - df_tech['bb_lower'])
        return df_tech

    def _extract_peak_structure_features(self, df_tech: pd.DataFrame, peaks_results: dict) -> pd.DataFrame:
        """提取峰值结构特征"""
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

    def _extract_momentum_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        """提取价格动量特征"""
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

    def _extract_volatility_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        """提取波动率结构特征"""
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

    def _extract_volume_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        """提取成交量特征"""
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

    def _extract_technical_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        """提取技术指标特征"""
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

    def _extract_microstructure_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        """提取市场微结构特征"""
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

    def _extract_temporal_features(self, df_tech: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
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

    def _create_target_variables(self, features_df: pd.DataFrame, df_tech: pd.DataFrame) -> pd.DataFrame:
        """创建目标变量"""

        # 前瞻收益率目标
        for period in [1, 3, 5, 10]:
            features_df[f'future_return_{period}'] = df_tech['close'].pct_change(period).shift(-period)

        # 分类目标：未来是否会出现显著价格移动
        threshold = df_tech['atr_14'].rolling(20).mean()
        features_df['significant_move_3'] = (abs(features_df['future_return_3']) > threshold / df_tech['close']).astype(
            int)
        features_df['significant_move_5'] = (abs(features_df['future_return_5']) > threshold / df_tech['close']).astype(
            int)

        # 方向目标
        features_df['direction_3'] = np.where(features_df['future_return_3'] > 0, 1,
                                              np.where(features_df['future_return_3'] < 0, -1, 0))
        features_df['direction_5'] = np.where(features_df['future_return_5'] > 0, 1,
                                              np.where(features_df['future_return_5'] < 0, -1, 0))

        return features_df

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


    # def _calculate_peak_trend(self, df_tech, window):
    #     """计算峰值趋势"""
    #     peak_prices = df_tech['close'].where(df_tech['is_peak'] == 1)
    #     return peak_prices.rolling(window=window, min_periods=2).apply(
    #         lambda x: np.polyfit(range(len(x.dropna())), x.dropna(), 1)[0] if len(x.dropna()) >= 2 else np.nan
    #     )

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


    # def _calculate_momentum_divergence(self, df_tech, period):
    #     """计算动量背离"""
    #     price_trend = df_tech['close'].rolling(period).apply(lambda x: stats.linregress(range(len(x)), x)[0])
    #     rsi_trend = df_tech['rsi_14'].rolling(period).apply(lambda x: stats.linregress(range(len(x)), x)[0])
    #
    #     # 背离信号：价格和RSI趋势方向相反
    #     return np.where((price_trend > 0) & (rsi_trend < 0), 1,  # 看跌背离
    #                     np.where((price_trend < 0) & (rsi_trend > 0), -1, 0))  # 看涨背离

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

    def _identify_volatility_regime(self, df_tech):
        """识别波动率状态"""
        vol_percentile = df_tech['atr_14'].rolling(50).rank(pct=True)
        return np.where(vol_percentile > 0.8, 2,  # 高波动
                        np.where(vol_percentile < 0.2, 0, 1))  # 低波动, 中等波动

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


class TradingSignalGenerator:
    """
    交易信号生成器 - 基于机器学习的华尔街级交易系统
    """

    def __init__(self):
        self.feature_engineer = PeakFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.is_trained = False

    def prepare_training_data(self, df: pd.DataFrame, peaks_results: dict):
        """准备训练数据"""
        features_df = self.feature_engineer.extract_comprehensive_features(df, peaks_results)

        # 分离特征和目标
        target_cols = [col for col in features_df.columns if col.startswith(('future_', 'significant_', 'direction_'))]
        feature_cols = [col for col in features_df.columns if col not in target_cols]

        X = features_df[feature_cols].copy()
        y_dict = {col: features_df[col].copy() for col in target_cols}

        print(f"📊 特征矩阵形状: {X.shape}")
        print(f"🎯 目标变量数量: {len(y_dict)}")
        print(f"🔍 特征数量: {len(feature_cols)}")

        return X, y_dict, feature_cols

    def train_models(self, X: pd.DataFrame, y_dict: dict, feature_cols: list):
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

    def generate_trading_signals(self, df: pd.DataFrame, peaks_results: dict) -> pd.DataFrame:
        """生成交易信号"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train_models()")

        print("🔮 生成交易信号...")

        # 提取特征
        features_df = self.feature_engineer.extract_comprehensive_features(df, peaks_results)
        X = features_df[self.feature_cols].copy()

        # 数据预处理
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X.fillna(0)),
            columns=X.columns,
            index=X.index
        )

        # 生成预测
        signals_df = pd.DataFrame(index=df.index)
        signals_df['timestamp'] = df.index
        signals_df['close_price'] = df['close']

        # 回归预测 - 预期收益率
        for target in ['future_return_3', 'future_return_5']:
            if target in self.models:
                predictions = self.models[target].predict(X_scaled)
                signals_df[f'predicted_{target}'] = predictions

        # 分类预测 - 方向和概率
        for target in ['direction_3', 'direction_5', 'significant_move_3', 'significant_move_5']:
            if target in self.models:
                predictions = self.models[target].predict(X_scaled)
                probabilities = self.models[target].predict_proba(X_scaled)

                signals_df[f'predicted_{target}'] = predictions

                # 获取最高概率
                max_prob = np.max(probabilities, axis=1)
                signals_df[f'{target}_confidence'] = max_prob

        # 生成综合交易信号
        signals_df = self._generate_composite_signals(signals_df)

        return signals_df

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


# 完整使用示例
def run_complete_trading_system(data_file_path: str, peak_file_path: str):
    """运行完整的交易信号系统"""
    df = load_process_data(data_file_path)
    peak_file_absolute_path = Path(build_data_dir() / peak_file_path)
    if not peak_file_absolute_path.exists():
        logBot.critical("Load peak data file not exist")
        return
    with Path(peak_file_absolute_path).open("r", encoding="utf-8") as f:
        peaks_results = json.load(f, object_hook=restore_hook)
    logBot.info("Load peak data Finish")
    signal_generator = TradingSignalGenerator()

    # 3. 准备训练数据
    X, y_dict, feature_cols = signal_generator.prepare_training_data(df, peaks_results)

    # 4. 训练模型
    models = signal_generator.train_models(X, y_dict, feature_cols)

    # 5. 生成交易信号
    signals_df = signal_generator.generate_trading_signals(df, peaks_results)

    # 6. 分析特征重要性
    feature_importance = signal_generator.analyze_feature_importance()

    # 7. 回测信号
    backtest_results = signal_generator.backtest_signals(signals_df)

    # 8. 输出最新信号
    print("\n🎯 最新交易信号 (最近10条):")
    latest_signals = signals_df.tail(10)[
        ['timestamp', 'close_price', 'trading_signal', 'signal_strength',
         'position_size', 'take_profit_pct', 'stop_loss_pct']
    ]

    for idx, row in latest_signals.iterrows():
        print(f"  {row['timestamp']}: {row['trading_signal']} | "
              f"强度: {row['signal_strength']:.1f} | "
              f"仓位: {row['position_size']:.2%} | "
              f"止盈: {row['take_profit_pct']:.2%} | "
              f"止损: {row['stop_loss_pct']:.2%}")

    return {
        'signals': signals_df,
        'models': models,
        'feature_importance': feature_importance,
        'backtest_results': backtest_results
    }


if __name__ == "__main__":
    data_file = "BTC_USDT_USDT-5m-futures.csv"
    peak_file = "peak_report_1756866531.json"

    try:
        results = run_complete_trading_system(data_file, peak_file)

        # 保存结果
        results['signals'].to_csv('btc_trading_signals.csv', index=False)
        print("💾 交易信号已保存至 btc_trading_signals.csv")

    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        print("请确保数据文件格式正确，并已导入必要的依赖模块")