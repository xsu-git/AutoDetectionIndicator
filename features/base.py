#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 15:55     Xsu         1.0         None
'''

from abc import ABC, abstractmethod
import pandas as pd


class ExtractFeatureBase(ABC):
    def __init__(self, name: str = None):
        """
        初始化基类

        参数:
        - name: 指标名称
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def extract(self, df_tech: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """
        抽象方法：计算技术指标
        所有子类必须实现此方法

        参数:
        - df_tech: 包含OHLCV数据的DataFrame
        - **kwargs: 其他参数

        返回:
        - 计算结果 (Series, DataFrame或Dict)
        """
        pass
