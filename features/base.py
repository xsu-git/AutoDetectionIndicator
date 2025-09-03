#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 15:55     Xsu         1.0         None
'''

from abc import ABC, abstractmethod
import pandas as pd
import os
import importlib
import inspect
from typing import List, Dict, Any, Type
import logging
from pathlib import Path
from utils.data_preprocess_util import build_feature_dir
from utils import logBot

class ExtractFeatureBase(ABC):
    def __init__(self, name: str = None):
        """
        初始化基类

        参数:
        - name: 指标名称
        """
        self.name = name or self.__class__.__name__
        self._data_cache = {}
        self._result_cache = {}

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


class FeatureExtractorLoader:
    """
    特征提取器自动加载和管理器
    """

    def __init__(self):
        self.features_dir = build_feature_dir()
        self.base_class = self._import_base_class()
        self.loaded_classes: Dict[str, Type] = {}
        self.extractors: Dict[str, ExtractFeatureBase] = {}
        # 自动加载所有特征提取器
        self.load_all_extractors()

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

    def discover_python_files(self) -> List[Path]:
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

    def load_module_from_file(self, file_path: Path) -> Any:
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

    def extract_extractor_classes(self, module: Any) -> List[Type]:
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

    def load_all_extractors(self) -> Dict[str, ExtractFeatureBase]:
        """
        加载所有特征提取器

        返回:
        - 特征提取器实例字典
        """
        logBot.info("Start Load Valid Features")
        python_files = self.discover_python_files()
        for file_path in python_files:
            module = self.load_module_from_file(file_path)
            if module is None:
                continue
            extractor_classes = self.extract_extractor_classes(module)
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

    def extract_all_features(self, df_tech: pd.DataFrame,
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
        if df_tech.empty:
            self.logger.warning("输入数据为空")
            return pd.DataFrame()

        # 确定要使用的提取器
        extractor_names = self.list_extractors()

        if include:
            extractor_names = [name for name in extractor_names if name in include]

        if exclude:
            extractor_names = [name for name in extractor_names if name not in exclude]

        self.logger.info(f"将使用 {len(extractor_names)} 个特征提取器")

        # 初始化结果DataFrame
        result_df = df_tech.copy()

        # 调用每个特征提取器
        for extractor_name in extractor_names:
            try:
                self.logger.debug(f"正在提取特征: {extractor_name}")

                extractor = self.extractors[extractor_name]
                features = extractor.extract(df_tech, **kwargs)

                if isinstance(features, pd.DataFrame):
                    # 合并特征到结果DataFrame
                    # 避免重复的索引列
                    feature_cols = [col for col in features.columns if col not in result_df.columns]
                    if feature_cols:
                        result_df = result_df.join(features[feature_cols], how='left')
                        self.logger.debug(f"{extractor_name} 添加了 {len(feature_cols)} 个特征")
                    else:
                        self.logger.warning(f"{extractor_name} 没有新特征添加")

                elif isinstance(features, pd.Series):
                    # 如果返回Series，作为单个特征列添加
                    feature_name = features.name or f"{extractor_name}_feature"
                    if feature_name not in result_df.columns:
                        result_df[feature_name] = features
                        self.logger.debug(f"{extractor_name} 添加了特征: {feature_name}")

                else:
                    self.logger.warning(f"{extractor_name} 返回了不支持的数据类型: {type(features)}")

            except Exception as e:
                self.logger.error(f"特征提取失败 {extractor_name}: {e}")
                continue

        self.logger.info(f"特征提取完成，总共 {len(result_df.columns)} 个特征列")
        return result_df

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

    def reload_extractors(self):
        """
        重新加载所有特征提取器（用于开发调试）
        """
        self.logger.info("重新加载所有特征提取器...")

        # 清空现有的
        self.loaded_classes.clear()
        self.extractors.clear()

        # 重新加载
        self.load_all_extractors()

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
                'module': extractor.__class__.__module__,
                'has_cache': bool(extractor._data_cache or extractor._result_cache)
            })

        return pd.DataFrame(info_data)


def main():
    """
    使用示例
    """
    # 创建加载器实例
    loader = FeatureExtractorLoader()

    # 生成示例数据
    import numpy as np
    np.random.seed(42)
    n = 100

    df_test = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.01) + np.random.rand(n),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.01) - np.random.rand(n),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.01),
        'volume': np.random.randint(1000, 10000, n)
    })

    print("=== 特征提取器加载器演示 ===")

    # 显示加载的提取器
    print(f"\n已加载的特征提取器: {loader.list_extractors()}")

    # 显示提取器详细信息
    print(f"\n提取器详细信息:")
    print(loader.get_extractor_info())

    # 提取所有特征
    print(f"\n开始提取所有特征...")
    all_features = loader.extract_all_features(df_test)
    print(f"原始特征数: {len(df_test.columns)}")
    print(f"提取后特征数: {len(all_features.columns)}")
    print(f"新增特征: {len(all_features.columns) - len(df_test.columns)}")

    # 提取特定特征
    if loader.extractors:
        first_extractor = list(loader.extractors.keys())[0]
        print(f"\n单独提取特征: {first_extractor}")
        single_feature = loader.extract_single_feature(first_extractor, df_test)
        print(f"结果形状: {single_feature.shape}")


if __name__ == "__main__":
    main()