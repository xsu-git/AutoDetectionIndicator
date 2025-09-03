#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/2 15:20     Xsu         1.0         None
'''
import os
import pandas as pd
from utils import logBot
from utils.data_preprocess_util import build_data_dir

def load_process_data(csv_file:str):
    '''
    :param csv_file:  .csv type file path to load
    :return:
    '''
    process_data_path = build_data_dir() / csv_file
    if not os.path.exists(process_data_path):
        logBot.critical("Load data failed: file not exist")
        return
    df = pd.read_csv(process_data_path)
    if 'date' in df.columns:
        # Handle time format: 2024-01-01T00:00:00.000000000+0000 to 2024-01-01T00:00:00.00+00
        df['timestamp'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)
    else:
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)
    # Rename columns to match detector requirements
    column_mapping = {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }
    if not all(col in df.columns for col in column_mapping.keys()):
        # Map OHLCV data by location.
        df_processed = pd.DataFrame({
            'timestamp': df['timestamp'],
            'open': pd.to_numeric(df.iloc[:, 1], errors='coerce'),
            'high': pd.to_numeric(df.iloc[:, 2], errors='coerce'),
            'low': pd.to_numeric(df.iloc[:, 3], errors='coerce'),
            'close': pd.to_numeric(df.iloc[:, 4], errors='coerce'),
            'volume': pd.to_numeric(df.iloc[:, 5], errors='coerce')
        })
    else:
        df_processed = df.rename(columns=column_mapping)
        df_processed['timestamp'] = df['timestamp']

    # Set timestamp as index
    df_processed = df_processed.set_index('timestamp')
    df_processed = df_processed.dropna()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    logBot.info(f"Load data shape: {df_processed.shape}")
    logBot.info(f"Load data Time Range:  {df_processed.index[0]} to {df_processed.index[-1]}")
    logBot.info(f"Load data Price Range:  {df_processed['close'].min():.2f} - {df_processed['close'].max():.2f}")
    logBot.info(f"Load data Finish")
    return df_processed

