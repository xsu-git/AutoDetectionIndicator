#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/2 13:40     Xsu         1.0         point detection
'''
import pandas as pd, numpy as np
from scipy.signal import argrelextrema

def label_local_lows(df: pd.DataFrame,
                     left=24, right=24,              # 对称窗口（例如1h数据=左右各24小时）
                     min_dd=0.05,                    # 从低点向前最高价的最小回撤比例(>=5%)
                     vol_z=1.0):                     # 成交量Z分数阈值（放量/恐慌）
    df = df.copy()
    # 局部最小：对close或low均可，这里取low更保守
    idx = argrelextrema(df['low'].values, np.less_equal, order=max(left, right))[0]
    lows = pd.Series(False, index=df.index)
    lows.iloc[idx] = True

    # 过滤：要求之前窗口内有明显回撤（跌到这个低点）
    # 计算每点往前left步的最高价
    rolling_max_pre = df['high'].rolling(left, min_periods=1).max().shift(1)
    dd = (rolling_max_pre - df['low']) / rolling_max_pre
    cond_dd = dd >= min_dd

    # 成交量相对放大（恐慌/换手）
    vol_zscore = (df['volume'] - df['volume'].rolling(240, min_periods=30).mean()) / df['volume'].rolling(240, min_periods=30).std()
    cond_vol = vol_zscore >= vol_z

    df['is_swing_low'] = lows & cond_dd & cond_vol.fillna(False)
    return df
