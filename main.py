#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time              @Author    @Version    @Desciption
---------------    -------    --------    -----------
2025/9/3 17:49     Xsu         1.0         None
'''
from process.loader import load_process_data,load_peak_data
from process.feature_detection import FeatureExtractorLoader
from utils.data_preprocess_util import feather_to_csv
from dotenv import load_dotenv
import os
from process.model_train import MachineLearnTrain

if __name__ == '__main__':
    # load_dotenv()
    # database_url = os.getenv('DATABASE_URL')

    data_file = "origin_data.csv"
    peak_file = "peak_report_1756977444.json"
    df_tech = load_process_data(data_file)
    peak_tech = load_peak_data(peak_file)

    features_df = FeatureExtractorLoader(df_tech,peak_tech).extract_all_features()
    target_cols = [col for col in features_df.columns if col.startswith(('future_', 'significant_', 'direction_'))]
    feature_cols = [col for col in features_df.columns if col not in target_cols]
    X = features_df[feature_cols].copy()
    y_dict = {col: features_df[col].copy() for col in target_cols}

    signal_generator = MachineLearnTrain()
    signal_generator.train(X,y_dict,feature_cols)

    signals_df = signal_generator.generate_trading_signals(df_tech, features_df)
    print(signals_df)
    # 6. 分析特征重要性
    feature_importance = signal_generator.analyze_feature_importance()
    #
    # # 7. 回测信号
    backtest_results = signal_generator.backtest_signals(signals_df)
    print(f"📊 特征矩阵形状: {X.shape}")
    print(f"🎯 目标变量数量: {len(y_dict)}")
    print(f"🔍 特征数量: {len(feature_cols)}")

