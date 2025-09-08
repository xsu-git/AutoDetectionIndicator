#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/2 13:40     Xsu         1.0         Format different types of files in a unified format
'''
import polars as pl
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import talib

TS_KEYS = {"timestamp"}  # need restore collection

def feather_to_csv(feather_file:str):
    '''
    :param feather_file:   .feather type file path
    :return:  convert to .csv file
    '''
    input_path = build_data_dir() / feather_file
    if not input_path.exists():
        raise Exception("Load data file not exist")
    output_path = build_data_dir() / (input_path.stem + ".csv")
    df = pl.read_ipc(input_path)
    df.write_csv(output_path)


def to_jsonable(o):
    if isinstance(o, pd.Timestamp):
        # ns → ms（整数）
        o = o.tz_convert('UTC') if o.tzinfo else o.tz_localize('UTC')
        return int(o.value // 1_000_000)
    if isinstance(o, datetime):
        o = o.astimezone(timezone.utc) if o.tzinfo else o.replace(tzinfo=timezone.utc)
        return int(o.timestamp() * 1000)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if not np.isfinite(v): return None
        return v
    raise TypeError


def restore_hook(d: dict):
    for k, v in list(d.items()):
        if k in TS_KEYS and isinstance(v, (int, float)):
            # 判定秒/毫秒：<1e11 多半是秒；否则按毫秒
            unit = "s" if v < 1e11 else "ms"
            d[k] = pd.to_datetime(int(v), unit=unit, utc=True)
    return d

def build_data_dir():
    return Path(__file__).parent.parent.absolute() / "data"

def build_feature_dir():
    return Path(__file__).parent.parent.absolute() / "features"

def calculate_technical_base(df: pd.DataFrame) -> pd.DataFrame:
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


if __name__ == '__main__':
    feather_to_csv("origin_data.feather")
