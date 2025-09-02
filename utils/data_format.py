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

def feather_to_csv(feather_file:str):
    '''
    :param feather_file:   .feather type file path
    :return:  convert to .csv file
    '''
    data_dir = Path(__file__).parent.parent.absolute() / "data"
    input_path = data_dir / feather_file
    if not os.path.exists(input_path):
        raise Exception("origin data file not exist")
    output_path = data_dir / (input_path.stem + ".csv")
    df = pl.read_ipc(input_path)
    df.write_csv(output_path)


if __name__ == '__main__':
    feather_to_csv("BTC_USDT_USDT-5m-futures.feather")