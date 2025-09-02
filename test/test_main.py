#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/2 14:59     Xsu         1.0         None
'''
from process.loader import load_process_data
from process.peak_detection import PeakDetector

# df = load_process_data("BTC_USDT_USDT-5m-futures.csv")
# detector = PeakDetector(
#         atr_period=14,                    # ATR周期
#         volatility_window=20,             # 波动率窗口
#         min_peak_distance=6,  # 最小峰值间距
#         smoothing_sigma=1.0   # 高斯平滑参数
#     )
#detector.detect_peaks_advanced(df)

