#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 17:57     Xsu         1.0         None
'''
import os
import importlib
import inspect
from typing import List, Dict, Any, Type
from pathlib import Path
from utils.data_preprocess_util import build_feature_dir,calculate_technical_base
from utils import logBot
from features.base import ExtractFeatureBase
import pandas as pd
import numpy as np

class FeatureExtractorLoader:
    """
    特征提取器自动加载和管理器
    """
    def __init__(self, df_tech: pd.DataFrame, peak_tech: Dict):
        self.features_dir = build_feature_dir()
        self.base_class = self._import_base_class()
        self.loaded_classes: Dict[str, Type] = {}
        self.df_tech = df_tech
        self.extractors: Dict[str, ExtractFeatureBase] = {}
        if self._verify_necessity(df_tech,peak_tech):
            self._load_all_extractors()

    def _verify_necessity(self,df_tech: pd.DataFrame, peak_tech: Dict) -> bool:
        verify_tag = False
        if df_tech.empty:
            logBot.critical("Invalid trade data input")
        elif peak_tech is None:
            logBot.critical("Invalid peak data input")
        else:
            verify_tag = True
            self.df_tech = df_tech
            self.peak_tech = peak_tech
        return verify_tag


    def _import_base_class(self) -> Type:
        """
        从features模块导入基类，确保类引用一致性
        """
        try:
            from features.base import ExtractFeatureBase
            return ExtractFeatureBase
        except ImportError:
            try:
                from features import ExtractFeatureBase
                return ExtractFeatureBase
            except ImportError:
                return ExtractFeatureBase

    def _discover_python_files(self) -> List[Path]:
        """
        find all file from features dir
        :return: .py files path
        """
        if not self.features_dir.exists():
            logBot.critical(f"Features目录不存在: {self.features_dir}")
            return []
        python_files = []
        for file_path in self.features_dir.rglob("*.py"):
            if file_path.name != "__init__.py" and not file_path.name.startswith("_"):
                python_files.append(file_path)
        return python_files

    def _load_module_from_file(self, file_path: Path) -> Any:
        """

        :param file_path: .py file path
        :return: success for load object
        """
        try:
            relative_path = file_path.relative_to(self.features_dir)
            module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
            full_module_name = f"{self.features_dir.name}.{module_name}"

            # 动态导入模块
            spec = importlib.util.spec_from_file_location(full_module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            logBot.debug(f"成功加载模块: {full_module_name}")
            return module

        except Exception as e:
            logBot.error(f"加载模块失败 {file_path}: {e}")
            return None

    def _extract_extractor_classes(self, module: Any) -> List[Type]:
        """
        从模块中提取继承自基类的特征提取器类

        参数:
        - module: 模块对象

        返回:
        - 特征提取器类列表
        """
        extractor_classes = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # 检查是否继承自基类且不是基类本身
            if (issubclass(obj, self.base_class) and
                    obj != self.base_class and
                    obj.__module__ == module.__name__):
                extractor_classes.append(obj)

        return extractor_classes

    def _load_all_extractors(self) -> Dict[str, ExtractFeatureBase]:
        """
        加载所有特征提取器

        返回:
        - 特征提取器实例字典
        """
        logBot.info("Start Load Valid Features")
        python_files = self._discover_python_files()
        for file_path in python_files:
            module = self._load_module_from_file(file_path)
            if module is None:
                continue
            extractor_classes = self._extract_extractor_classes(module)
            for extractor_class in extractor_classes:
                try:
                    extractor_instance = extractor_class()
                    class_name = extractor_class.__name__
                    self.loaded_classes[class_name] = extractor_class
                    self.extractors[class_name] = extractor_instance
                    logBot.info(f"Success Load feature_class: {class_name}")
                except Exception as e:
                    logBot.error(f"Failed Load feature_class: {extractor_class.__name__}: {e}")
        return self.extractors

    def get_extractor(self, name: str) -> ExtractFeatureBase:
        """
        根据名称获取特征提取器实例

        参数:
        - name: 提取器名称

        返回:
        - 特征提取器实例
        """
        return self.extractors.get(name)

    def list_extractors(self) -> List[str]:
        return list(self.extractors.keys())

    def extract_all_features(self,
                             include: List[str] = None,
                             exclude: List[str] = None,
                             **kwargs) -> pd.DataFrame:
        """
        调用所有特征提取器的extract方法

        参数:
        - df_tech: 输入数据
        - include: 仅包含的提取器名称列表
        - exclude: 排除的提取器名称列表
        - **kwargs: 传递给extract方法的额外参数

        返回:
        - 包含所有特征的DataFrame
        """
        logBot.info("Start Extraction feature engineering...")
        extractor_names = self.list_extractors()
        if include:
            extractor_names = [name for name in extractor_names if name in include]
        if exclude:
            extractor_names = [name for name in extractor_names if name not in exclude]
        if extractor_names is None:
            logBot.critical("Invalid feature find")
            return pd.DataFrame()
        self.df_tech = calculate_technical_base(self.df_tech)
        logBot.info("Success preprocess trade data")
        self.df_tech['is_peak'] = 0
        self.df_tech['peak_type'] = 0  # -1: low, 0: none, 1: high
        self.df_tech['peak_score'] = 0.0

        # 标记高低点
        for high in self.peak_tech['highs']:
            idx = high['index']
            if idx < len(self.df_tech):
                self.df_tech.iloc[idx, self.df_tech.columns.get_loc('is_peak')] = 1
                self.df_tech.iloc[idx, self.df_tech.columns.get_loc('peak_type')] = 1
                self.df_tech.iloc[idx, self.df_tech.columns.get_loc('peak_score')] = high['score']

        for low in self.peak_tech['lows']:
            idx = low['index']
            if idx < len(self.df_tech):
                self.df_tech.iloc[idx, self.df_tech.columns.get_loc('is_peak')] = 1
                self.df_tech.iloc[idx, self.df_tech.columns.get_loc('peak_type')] = -1
                self.df_tech.iloc[idx, self.df_tech.columns.get_loc('peak_score')] = low['score']

        # 提取各类特征
        features_df = pd.DataFrame(index=self.df_tech.index)

        for extractor_name in extractor_names:
            try:
                extractor = self.extractors[extractor_name]
                logBot.debug(f"Start extract {extractor.name} features")
                features = extractor.extract(self.df_tech, **kwargs)
                features_df = pd.concat([features_df, features], axis=1)
            except Exception as e:
                logBot.error(f"特征提取失败 {extractor_name}: {e}")
                continue

        features_df = self._create_target_variables(features_df, self.df_tech)
        logBot.info(f"Success Extract All valid features: Total {len(features_df.columns)} special conquest column")
        return features_df.dropna()

    def extract_single_feature(self, extractor_name: str, df_tech: pd.DataFrame,
                               **kwargs) -> pd.DataFrame:
        """
        调用单个特征提取器

        参数:
        - extractor_name: 提取器名称
        - df_tech: 输入数据
        - **kwargs: 额外参数

        返回:
        - 特征DataFrame
        """
        if extractor_name not in self.extractors:
            raise ValueError(f"特征提取器不存在: {extractor_name}")

        extractor = self.extractors[extractor_name]
        return extractor.extract(df_tech, **kwargs)

    def get_extractor_info(self) -> pd.DataFrame:
        """
        获取所有特征提取器的信息

        返回:
        - 包含提取器信息的DataFrame
        """
        info_data = []

        for name, extractor in self.extractors.items():
            info_data.append({
                'name': name,
                'class_name': extractor.__class__.__name__,
                'module': extractor.__class__.__module__
            })

        return pd.DataFrame(info_data)


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

def main():
    """
    使用示例
    """
    # 创建加载器实例
    loader = FeatureExtractorLoader("")

    # 生成示例数据
    # import numpy as np
    # np.random.seed(42)
    # n = 100
    #
    # df_test = pd.DataFrame({
    #     'open': 100 + np.cumsum(np.random.randn(n) * 0.01),
    #     'high': 100 + np.cumsum(np.random.randn(n) * 0.01) + np.random.rand(n),
    #     'low': 100 + np.cumsum(np.random.randn(n) * 0.01) - np.random.rand(n),
    #     'close': 100 + np.cumsum(np.random.randn(n) * 0.01),
    #     'volume': np.random.randint(1000, 10000, n)
    # })
    #
    # print("=== 特征提取器加载器演示 ===")
    #
    # # 显示加载的提取器
    # print(f"\n已加载的特征提取器: {loader.list_extractors()}")
    #
    # # 显示提取器详细信息
    # print(f"\n提取器详细信息:")
    # print(loader.get_extractor_info())
    #
    # # 提取所有特征
    # print(f"\n开始提取所有特征...")
    # all_features = loader.extract_all_features(df_test)
    # print(f"原始特征数: {len(df_test.columns)}")
    # print(f"提取后特征数: {len(all_features.columns)}")
    # print(f"新增特征: {len(all_features.columns) - len(df_test.columns)}")
    #
    # # 提取特定特征
    # if loader.extractors:
    #     first_extractor = list(loader.extractors.keys())[0]
    #     print(f"\n单独提取特征: {first_extractor}")
    #     single_feature = loader.extract_single_feature(first_extractor, df_test)
    #     print(f"结果形状: {single_feature.shape}")


if __name__ == "__main__":
    main()