#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/2 13:40     Xsu         1.0         Format different types of files in a unified format
'''
import polars as pl
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

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


if __name__ == '__main__':
    feather_to_csv("BTC_USDT_USDT-5m-futures.feather")
    # load_process_data("BTC_USDT_USDT-5m-futures.csv")