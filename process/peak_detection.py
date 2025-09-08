#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/2 13:40     Xsu         1.0         point detection
'''

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import gaussian_filter1d
import talib
from typing import Dict, List
import warnings
from utils import logBot
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time,json
from process.loader import load_process_data
from utils.data_preprocess_util import to_jsonable,build_data_dir


warnings.filterwarnings('ignore')


class PeakDetector:
    """
    """

    def __init__(self,
                 atr_period: int = 14,
                 volatility_window: int = 20,
                 min_peak_distance: int = 5,
                 smoothing_sigma: float = 1.0,
                 is_report: bool = True,
                 is_plot: bool = False,
                 is_reserve: bool = True):
        """
        Init Args:
            atr_period: ATR calculation period
            volatility_window: Volatility calculation window
            min_peak_distance: Minimum peak spacing
            smoothing_sigma: Gaussian smoothing parameter
        """
        self.atr_period = atr_period
        self.volatility_window = volatility_window
        self.min_peak_distance = min_peak_distance
        self.smoothing_sigma = smoothing_sigma
        self.is_report = is_report
        self.is_plot = is_plot
        self.is_reserve = is_reserve

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Calculation of basic technical indicators
        :param df:
        :return:
        '''
        df = df.copy()
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Price Momentum
        df['momentum'] = df['close'].pct_change(periods=10)

        # Volume-weighted average price
        if 'volume' in df.columns:
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

        return df

    def _calculate_dynamic_thresholds(self, df: pd.DataFrame) -> Dict:
        """
        Calculating dynamic threshold parameters
        :param df:
        :return:
        """

        # Price volatility
        price_volatility = df['close'].rolling(self.volatility_window).std()
        avg_volatility = price_volatility.mean()

        # ATR basic threshold
        avg_atr = df['atr'].mean()

        # Dynamic minimum altitude threshold (based on 0.3-0.8 times ATR)
        min_height_pct = np.clip(avg_atr / df['close'].mean() * 0.5, 0.001, 0.01)

        # 动态距离参数
        volatility_factor = avg_volatility / df['close'].mean()
        dynamic_distance = max(3, int(self.min_peak_distance * (1 + volatility_factor * 10)))

        return {
            'min_height_pct': min_height_pct,
            'dynamic_distance': min(dynamic_distance, 20),  # 最大不超过20
            'avg_atr': avg_atr,
            'volatility_factor': volatility_factor
        }

    def detect_peaks_advanced(self, df: pd.DataFrame) -> Dict:
        """
        高级峰值检测算法 - 精准版

        核心策略：
        1. 多重价格序列组合验证（close, high/low, hl2）
        2. 动态阈值自适应调整
        3. 多算法融合确认
        4. 专业评分过滤
        """
        df_tech = self._calculate_technical_indicators(df)
        thresholds = self._calculate_dynamic_thresholds(df_tech)

        # 核心价格序列
        close_prices = df_tech['close'].values
        high_prices = df_tech['high'].values
        low_prices = df_tech['low'].values
        hl2_prices = (df_tech['high'] + df_tech['low']) / 2

        # 应用高斯平滑降噪
        if self.smoothing_sigma > 0:
            close_smooth = gaussian_filter1d(close_prices, sigma=self.smoothing_sigma)
            hl2_smooth = gaussian_filter1d(hl2_prices.values, sigma=self.smoothing_sigma)
        else:
            close_smooth = close_prices
            hl2_smooth = hl2_prices.values

        # 检测高点 - 使用三重验证
        peaks_high = self._detect_highs(
            close_smooth, high_prices, hl2_smooth, df_tech, thresholds
        )

        # 检测低点 - 使用三重验证
        peaks_low = self._detect_lows(
            close_smooth, low_prices, hl2_smooth, df_tech, thresholds
        )

        # 专业验证和评分
        validated_highs = self._validate_peaks(peaks_high, df_tech, 'high')
        validated_lows = self._validate_peaks(peaks_low, df_tech, 'low')

        reports = {
            'highs': validated_highs,
            'lows': validated_lows,
            'detection_params': {
                'total_candles': len(df_tech),
                'atr_threshold': thresholds['avg_atr'],
                'volatility_factor': thresholds['volatility_factor'],
                'dynamic_distance': thresholds['dynamic_distance']
            }
        }
        self._peak_reserve(reports)

        if self.is_report:
            self._peak_report(reports)
        if self.is_plot:
            self._peak_plot(df,reports)
        return reports

    def _detect_highs(self, close_smooth: np.ndarray, high_series: np.ndarray,
                      hl2_smooth: np.ndarray, df_tech: pd.DataFrame, thresholds: Dict) -> np.ndarray:
        """
        精准高点检测 - 三重算法验证

        策略组合：
        1. 平滑收盘价find_peaks检测
        2. 原始高价argrelextrema检测  
        3. HL2平滑序列峰值检测
        4. 布林带上轨突破确认
        """

        # 算法1: 平滑收盘价峰值检测（主要方法）
        peaks1, properties1 = find_peaks(
            close_smooth,
            height=np.percentile(close_smooth, 65),
            distance=thresholds['dynamic_distance'],
            prominence=thresholds['avg_atr'] * 0.4,
            width=(2, 15)  # 峰值宽度控制
        )

        # 算法2: 高价序列相对极值（辅助验证）
        peaks2 = argrelextrema(high_series, np.greater, order=4)[0]

        # 算法3: HL2平滑序列峰值（噪音过滤）
        peaks3, _ = find_peaks(
            hl2_smooth,
            height=np.percentile(hl2_smooth, 62),
            distance=max(3, thresholds['dynamic_distance'] - 2),
            prominence=thresholds['avg_atr'] * 0.35
        )

        # 算法4: 布林带突破高点（关键阻力）
        bb_upper = df_tech['bb_upper'].values
        price_above_bb = close_smooth > bb_upper
        bb_breakthrough = []

        for i in range(1, len(price_above_bb) - 1):
            if (price_above_bb[i] and not price_above_bb[i - 1] and
                    close_smooth[i] > close_smooth[i - 1] and close_smooth[i] > close_smooth[i + 1]):
                bb_breakthrough.append(i)

        # 融合所有检测结果
        all_candidates = np.concatenate([peaks1, peaks2, peaks3, bb_breakthrough])
        unique_peaks = np.unique(all_candidates)

        # 边界过滤 - 确保有足够的前后数据进行验证
        boundary_buffer = max(10, thresholds['dynamic_distance'])
        valid_peaks = unique_peaks[
            (unique_peaks >= boundary_buffer) &
            (unique_peaks < len(close_smooth) - boundary_buffer)
            ]

        return valid_peaks

    def _detect_lows(self, close_smooth: np.ndarray, low_series: np.ndarray,
                     hl2_smooth: np.ndarray, df_tech: pd.DataFrame, thresholds: Dict) -> np.ndarray:
        """
        精准低点检测 - 三重算法验证

        策略组合：
        1. 反转收盘价find_peaks检测
        2. 原始低价argrelextrema检测
        3. HL2平滑序列谷值检测  
        4. 布林带下轨突破确认
        """

        # 算法1: 反转平滑收盘价检测谷值（主要方法）
        inverted_close = -close_smooth
        peaks1, _ = find_peaks(
            inverted_close,
            height=-np.percentile(close_smooth, 35),
            distance=thresholds['dynamic_distance'],
            prominence=thresholds['avg_atr'] * 0.4,
            width=(2, 15)
        )

        # 算法2: 低价序列相对极小值（辅助验证）
        peaks2 = argrelextrema(low_series, np.less, order=4)[0]

        # 算法3: HL2平滑序列谷值（噪音过滤）
        inverted_hl2 = -hl2_smooth
        peaks3, _ = find_peaks(
            inverted_hl2,
            height=-np.percentile(hl2_smooth, 38),
            distance=max(3, thresholds['dynamic_distance'] - 2),
            prominence=thresholds['avg_atr'] * 0.35
        )

        # 算法4: 布林带下轨突破低点（关键支撑）
        bb_lower = df_tech['bb_lower'].values
        price_below_bb = close_smooth < bb_lower
        bb_breakthrough = []

        for i in range(1, len(price_below_bb) - 1):
            if (price_below_bb[i] and not price_below_bb[i - 1] and
                    close_smooth[i] < close_smooth[i - 1] and close_smooth[i] < close_smooth[i + 1]):
                bb_breakthrough.append(i)

        # 融合所有检测结果
        all_candidates = np.concatenate([peaks1, peaks2, peaks3, bb_breakthrough])
        unique_peaks = np.unique(all_candidates)

        # 边界过滤
        boundary_buffer = max(10, thresholds['dynamic_distance'])
        valid_peaks = unique_peaks[
            (unique_peaks >= boundary_buffer) &
            (unique_peaks < len(close_smooth) - boundary_buffer)
            ]

        return valid_peaks

    def _validate_peaks(self, peaks: np.ndarray, df_tech: pd.DataFrame,
                        peak_type: str) -> List[Dict]:
        """验证和评分峰值"""
        validated_peaks = []

        if len(peaks) == 0:
            return validated_peaks

        close_prices = df_tech['close'].values

        logBot.info(f"Processing data validate {peak_type} peaks progress: ")

        for peak_idx in tqdm(peaks, desc="Processing",total=len(peaks),mininterval=0.3, miniters=500, smoothing=0.1):
            if peak_idx >= len(close_prices):
                continue
            # 计算峰值强度评分
            score = self._calculate_peak_score(peak_idx, df_tech, peak_type)

            # 最低分数阈值
            min_score = 0.3 if peak_type == 'high' else 0.3

            if score >= min_score:
                validated_peaks.append({
                    'index': int(peak_idx),
                    'price': float(close_prices[peak_idx]),
                    'timestamp': df_tech.index[peak_idx] if hasattr(df_tech.index, '__getitem__') else peak_idx,
                    'score': float(score),
                    'type': peak_type
                })



        # 按分数排序
        validated_peaks.sort(key=lambda x: x['score'], reverse=True)

        # 去除距离过近的低分峰值
        filtered_peaks = self._filter_nearby_peaks(validated_peaks)

        return filtered_peaks

    def _calculate_peak_score(self, peak_idx: int, df_tech: pd.DataFrame,
                              peak_type: str) -> float:
        """计算峰值评分"""
        try:
            score = 0.0
            close_prices = df_tech['close'].values
            peak_price = close_prices[peak_idx]

            # 1. 局部极值强度 (权重: 0.25)
            window = 5
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(close_prices), peak_idx + window + 1)
            local_prices = close_prices[start_idx:end_idx]

            if peak_type == 'high':
                local_strength = (peak_price - np.min(local_prices)) / (
                            np.max(local_prices) - np.min(local_prices) + 1e-8)
            else:
                local_strength = (np.max(local_prices) - peak_price) / (
                            np.max(local_prices) - np.min(local_prices) + 1e-8)

            score += local_strength * 0.25

            # 2. 成交量确认 (权重: 0.2)
            if 'volume' in df_tech.columns:
                volume = df_tech['volume'].iloc[peak_idx]
                avg_volume = df_tech['volume'].rolling(20).mean().iloc[peak_idx]
                if not pd.isna(avg_volume) and avg_volume > 0:
                    volume_score = min(volume / avg_volume, 3.0) / 3.0
                    score += volume_score * 0.2

            # 3. 技术指标确认 (权重: 0.25)
            rsi = df_tech['rsi'].iloc[peak_idx]
            if not pd.isna(rsi):
                if peak_type == 'high':
                    rsi_score = max(0, (rsi - 70) / 30) if rsi > 70 else 0
                else:
                    rsi_score = max(0, (30 - rsi) / 30) if rsi < 30 else 0
                score += rsi_score * 0.25

            # 4. 价格位置评分 (权重: 0.15)
            sma_20 = df_tech['sma_20'].iloc[peak_idx]
            if not pd.isna(sma_20) and sma_20 > 0:
                price_position = abs(peak_price - sma_20) / sma_20
                position_score = min(price_position / 0.02, 1.0)  # 2%以上偏离得满分
                score += position_score * 0.15

            # 5. ATR标准化强度 (权重: 0.15)
            atr = df_tech['atr'].iloc[peak_idx]
            if not pd.isna(atr) and atr > 0:
                price_move = abs(peak_price - close_prices[max(0, peak_idx - 1)])
                atr_score = min(price_move / atr, 2.0) / 2.0
                score += atr_score * 0.15

            return min(score, 1.0)

        except Exception as e:
            return 0.0

    def _filter_nearby_peaks(self, peaks: List[Dict]) -> List[Dict]:
        '''
        Filter peaks that are too close
        :param peaks:
        :return:
        '''
        if len(peaks) <= 1:
            return peaks

        filtered = []
        min_distance = self.min_peak_distance

        for peak in peaks:
            is_valid = True
            for existing in filtered:
                if abs(peak['index'] - existing['index']) < min_distance:
                    # Retention score higher
                    if peak['score'] > existing['score']:
                        filtered.remove(existing)
                    else:
                        is_valid = False
                    break

            if is_valid:
                filtered.append(peak)

        return sorted(filtered, key=lambda x: x['index'])

    def _peak_report(self, report: dict):
        '''
        :param report: peak result
        :return:
        '''
        logBot.info("====== Accurate peak detection ======")
        logBot.info(f"Total number of K lines: {report['detection_params']['total_candles']}")
        logBot.info(f"ATR threshold: {report['detection_params']['atr_threshold']:.2f}")
        logBot.info(f"Dynamic Spacing: {report['detection_params']['dynamic_distance']}")
        logBot.info(f"Detected: {len(report['highs'])} high points")
        logBot.info(f"Detected: {len(report['lows'])} low points")

        # Show high quality peaks (score > 0.6)
        high_quality_highs = [h for h in report['highs'] if h['score'] > 0.6]
        high_quality_lows = [l for l in report['lows'] if l['score'] > 0.6]
        logBot.info(f"High quality high points (score > 0.6): {len(high_quality_highs)}")
        logBot.info(f"High quality low points (score > 0.6): {len(high_quality_lows)}")

    def _peak_plot(self, df: pd.DataFrame, report: dict):
        """绘制峰值检测结果"""

        plt.figure(figsize=(15, 10))

        # 子图1: 价格走势 + 峰值标记
        plt.subplot(2, 1, 1)

        # 绘制K线收盘价
        plt.plot(df.index, df['close'], 'b-', linewidth=1, alpha=0.7, label='Price')

        # 标记高点
        if report['highs']:
            high_indices = [h['index'] for h in report['highs']]
            high_prices = [h['price'] for h in report['highs']]
            high_scores = [h['score'] for h in report['highs']]

            # 根据评分设置颜色
            colors = ['red' if score > 0.7 else 'orange' if score > 0.5 else 'pink' for score in high_scores]
            sizes = [100 * score for score in high_scores]

            plt.scatter(df.index[high_indices], high_prices,
                        c=colors, s=sizes, marker='v', alpha=0.8,
                        label=f'High point ({len(report["highs"])})', zorder=5)

        # 标记低点
        if report['lows']:
            low_indices = [l['index'] for l in report['lows']]
            low_prices = [l['price'] for l in report['lows']]
            low_scores = [l['score'] for l in report['lows']]

            # 根据评分设置颜色
            colors = ['green' if score > 0.7 else 'lightgreen' if score > 0.5 else 'lightblue' for score in low_scores]
            sizes = [100 * score for score in low_scores]

            plt.scatter(df.index[low_indices], low_prices,
                        c=colors, s=sizes, marker='^', alpha=0.8,
                        label=f'Low point ({len(report["lows"])})', zorder=5)

        plt.title('K-line peak detection results', fontsize=14, fontweight='bold')
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.bar(df.index, df['volume'], alpha=0.6, color='gray', width=pd.Timedelta(minutes=4))
        plt.title('Volume', fontsize=12)
        plt.ylabel('Volume', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _peak_reserve(self, content:dict):
        output_path = build_data_dir() / f"peak_report_{str(int(time.time()))}.json"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False, indent=2,default=to_jsonable, allow_nan=False))

if __name__ == '__main__':
    df = load_process_data("origin_data.csv")
    detector = PeakDetector(
        atr_period=14,  # ATR周期
        volatility_window=20,  # 波动率窗口
        min_peak_distance=6,  # 最小峰值间距
        smoothing_sigma=1.0  # 高斯平滑参数
    )
    detector.detect_peaks_advanced(df)