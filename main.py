#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 17:49     Xsu         1.0         None
'''
from process.loader import load_process_data,load_peak_data
from process.feature_detection import FeatureExtractorLoader

if __name__ == '__main__':
    data_file = "origin_data.csv"
    peak_file = "peak_report_1756866531.json"
    s = load_process_data(data_file)
    b = load_peak_data(peak_file)

    features_df = FeatureExtractorLoader(s,b).extract_all_features()
    target_cols = [col for col in features_df.columns if col.startswith(('future_', 'significant_', 'direction_'))]
    feature_cols = [col for col in features_df.columns if col not in target_cols]
    X = features_df[feature_cols].copy()
    y_dict = {col: features_df[col].copy() for col in target_cols}

    print(f"📊 特征矩阵形状: {X.shape}")
    print(f"🎯 目标变量数量: {len(y_dict)}")
    print(f"🔍 特征数量: {len(feature_cols)}")

